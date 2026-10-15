"""Brick U3 -- "Show what the tool sees".

The wall signal the cut followed is the one input to a Reshape the user
cannot see.  After a Preview, one tick puts it up as an ordinary Image
layer, ``guide_walls``, built from ``ReshapeResult.elevation`` -- an ROI
crop the engine already retains -- placed at the ROI offset in a
volume-shaped float32 array so it lines up with the labels and with the
markers on the Refine layer.

Two halves, tested in two halves:

* **the panel** owns the tick, its enabled rule and the legend, which is
  worded from the WALL KIND because the same greyscale means two
  different things -- an image-derived elevation is evidence of an edge,
  a shape-derived one is the object's own thickness;
* **the orchestrator** owns the layer: the ROI placement, the byte
  budget, the rebuild on a re-Preview, and the removal at every one of
  the six exits.

The layer rides the existing temp-layer registry, so it
is unmanaged, ``_epicycle_temporary``, read-only, and hidden before it is
removed.  It adds NO retention: the elevation is already held by
``_reshape_result``.

Qt for the panel half (offscreen), Qt-free fakes for the orchestrator
half -- the real methods bound with ``__get__``, so these exercise the
actual code rather than a paraphrase of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from .test_reshape_split_orchestration import (
    FakeAdv,
    FakeAnnotator,
    bind,
)


def _K():
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    return Curation3DAnnotator


def _legend_shown(widget) -> bool:
    """Is the legend on screen WITHIN its card?

    ``isVisibleTo(widget)`` is the wrong question: S5 itself is hidden
    outside review, so it answers False for every state and the
    assertion would pass by accident.  The legend's own rule is relative
    to the card it lives in.
    """
    return bool(
        widget.split_walls_legend.isVisibleTo(widget._split_result_container)
    )


def _M():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


def _mod():
    """The ORCHESTRATOR module -- its constants, and the `Image` /
    `Labels` names the temporary layer is built through."""
    from napari_epicycle.annotator import curation_3d_widget as M

    return M


# --------------------------------------------------------------------------- #
# the panel half
# --------------------------------------------------------------------------- #
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


class TestTheTickExists:
    def test_it_is_a_parented_checkbox_with_the_owner_wording(self, widget):
        from qtpy.QtWidgets import QCheckBox

        cb = widget.split_show_walls_cb
        assert isinstance(cb, QCheckBox)
        assert cb.text() == "Show what the tool sees"
        assert cb.parentWidget() is not None, "every QWidget has a parent"

    def test_it_lives_in_the_result_card(self, widget):
        """S5, not S4: the pinned strip's height is asserted constant
        across states, so a tick there would grow EVERY state including
        idle.  S5 exists only in review, which is exactly when a wall
        picture is meaningful."""
        node, found = widget.split_show_walls_cb, False
        while node is not None:
            if node is widget._split_result_container:
                found = True
                break
            node = node.parentWidget()
        assert found, "the tick belongs to the RESULT card"

    def test_it_shares_the_card_header_row(self, widget):
        """It costs the page NOTHING: review-with-markers-open has 6 px
        of its ceiling left (UX round 1.4), and a stacked row is 15.  It
        rides the header row instead, which is already there."""
        from qtpy.QtWidgets import QLabel

        lay = widget._split_result_container.layout()
        row = lay.itemAt(0).layout()
        assert row is not None, "the header is a ROW now, not a bare label"
        widgets = [row.itemAt(i).widget() for i in range(row.count())]
        widgets = [w for w in widgets if w is not None]
        assert widget.split_show_walls_cb in widgets
        assert any(isinstance(w, QLabel) and w.text() == "RESULT"
                   for w in widgets), "the card keeps its title"

    def test_it_starts_disabled_and_unticked(self, widget):
        assert widget.split_show_walls_cb.isEnabled() is False
        assert widget.split_show_walls_cb.isChecked() is False

    def test_the_disabled_tooltip_says_what_to_do(self, widget):
        tip = widget.split_show_walls_cb.toolTip()
        assert "Preview" in tip
        tip.encode("cp1252")


class TestTheEnabledRule:
    def test_enabling_it_swaps_the_tooltip_for_what_it_does(self, widget):
        widget.set_walls_toggle_enabled(True)
        assert widget.split_show_walls_cb.isEnabled() is True
        tip = widget.split_show_walls_cb.toolTip()
        assert "layer" in tip.lower()

    def test_update_split_ui_state_can_enable_it(self, widget):
        widget.update_split_ui_state(walls_toggle_enabled=True)
        assert widget.split_show_walls_cb.isEnabled() is True

    def test_leaving_review_unticks_AND_disables_it(self, widget):
        widget.set_walls_toggle_enabled(True)
        widget.split_show_walls_cb.setChecked(True)
        widget.update_split_ui_state(review_active=False)
        assert widget.split_show_walls_cb.isChecked() is False
        assert widget.split_show_walls_cb.isEnabled() is False
        assert _legend_shown(widget) is False

    def test_clear_split_guided_ui_clears_it_too(self, widget):
        widget.set_walls_toggle_enabled(True)
        widget.split_show_walls_cb.setChecked(True)
        widget.clear_split_guided_ui()
        assert widget.split_show_walls_cb.isChecked() is False
        assert widget.split_show_walls_cb.isEnabled() is False

    def test_the_programmatic_untick_does_not_fire_the_callback(self, widget):
        """Screen and viewer are kept in step by the ORCHESTRATOR, which
        removes the layer explicitly at each exit.  A panel untick that
        also fired the callback would make the order load-bearing."""
        fired = []
        widget.set_split_callbacks(on_show_walls_toggled=fired.append)
        widget.set_walls_toggle_enabled(True)
        widget.split_show_walls_cb.setChecked(True)
        assert fired == [True]
        widget.clear_walls_toggle()
        assert fired == [True], "a programmatic clear is silent"


class TestTheLegend:
    def test_it_is_hidden_until_the_box_is_ticked(self, widget):
        assert _legend_shown(widget) is False
        widget.set_walls_toggle_enabled(True)
        widget.split_show_walls_cb.setChecked(True)
        assert _legend_shown(widget) is True
        widget.split_show_walls_cb.setChecked(False)
        assert _legend_shown(widget) is False

    def test_an_image_wall_says_where_an_edge_is(self, widget):
        M = _M()
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        assert widget.walls_legend_text() == M.WALLS_LEGEND_IMAGE
        assert "edge" in M.WALLS_LEGEND_IMAGE

    def test_a_pinch_wall_says_bright_is_THIN(self, widget):
        """``distance`` is ``norm(-EDT(region))`` (``walls.py``): bright
        where the EDT is LOW, i.e. where the object is THIN.

        The sentence this replaces -- "This is the object's own
        thickness" -- is the picture upside down, and it is the caption
        under the one image the user is looking at to decide whether the
        tool saw the waist.  They would have gone looking for the bright
        band in the wrong half of the object.
        """
        M = _M()
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_SHAPE)
        assert widget.walls_legend_text() == M.WALLS_LEGEND_PINCH
        assert "thin" in M.WALLS_LEGEND_PINCH.lower()
        assert "object" in M.WALLS_LEGEND_PINCH
        assert "cell" not in M.WALLS_LEGEND_PINCH.lower()

    def test_between_the_spines_says_the_SPINES_ARE_DARK(self, widget):
        """``skeleton`` is ``norm(EDT(~skeleton))``: bright FAR FROM the
        medial axis.  The two merged objects' spines are the DARK lines
        and the bright sheet between them is where the cut goes -- the
        opposite reading of the same greyscale from the pinch wall."""
        M = _M()
        idx = M.WALL_KINDS.index("skeleton")
        widget.split_walls_from_combo.setCurrentIndex(idx)
        assert widget.walls_legend_text() == M.WALLS_LEGEND_SPINES
        assert "dark" in M.WALLS_LEGEND_SPINES.lower()
        assert "object" in M.WALLS_LEGEND_SPINES
        assert "cell" not in M.WALLS_LEGEND_SPINES.lower()

    def test_no_wall_says_there_is_nothing_to_see(self, widget):
        """``none`` is literally ``zeros``.  A user told "bright = thin"
        here would hunt for a bright band that cannot exist."""
        M = _M()
        idx = M.WALL_KINDS.index("none")
        widget.split_walls_from_combo.setCurrentIndex(idx)
        assert widget.walls_legend_text() == M.WALLS_LEGEND_FLAT
        assert "midline" in M.WALLS_LEGEND_FLAT.lower()

    def test_the_three_shape_sentences_are_all_different(self):
        """The bug in one line: two of these used to be the same
        sentence, and it was wrong for both."""
        M = _M()
        assert len({M.WALLS_LEGEND_PINCH, M.WALLS_LEGEND_SPINES,
                    M.WALLS_LEGEND_FLAT, M.WALLS_LEGEND_IMAGE}) == 4

    @pytest.mark.parametrize("idx", range(9))
    def test_every_wall_kind_gets_the_sentence_its_physics_earns(
            self, widget, idx):
        M = _M()
        widget.split_walls_from_combo.setCurrentIndex(idx)
        text = widget.walls_legend_text()
        kind = M.WALL_KINDS[idx]
        if idx in M.IMAGE_DERIVED_WALL_INDICES:
            expected = M.WALLS_LEGEND_IMAGE
        elif kind == "skeleton":
            expected = M.WALLS_LEGEND_SPINES
        elif kind == "none":
            expected = M.WALLS_LEGEND_FLAT
        else:
            expected = M.WALLS_LEGEND_PINCH
        assert text == expected, "{} got {!r}".format(kind, text)

    def test_the_geometry_only_kinds_are_the_three_that_are_keyed(self):
        """A guard against the engine growing a fourth geometry wall and
        quietly inheriting the pinch sentence: it would be caught here
        before a user reads it."""
        from packages.epicycle_annotator.reshape import GEOMETRY_ONLY_WALLS

        M = _M()
        geometry = {k for k in M.WALL_KINDS
                    if M.WALL_KINDS.index(k) not in
                    M.IMAGE_DERIVED_WALL_INDICES}
        assert geometry == {"distance", "skeleton", "none"}
        assert geometry <= set(GEOMETRY_ONLY_WALLS)

    def test_the_legend_swaps_with_the_wall_kind_while_it_is_up(self, widget):
        M = _M()
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        widget.set_walls_toggle_enabled(True)
        widget.split_show_walls_cb.setChecked(True)
        assert widget.split_walls_legend.text() == M.WALLS_LEGEND_IMAGE

    def test_all_four_sentences_are_cp1252_safe_and_end_in_a_full_stop(self):
        M = _M()
        for text in (M.WALLS_LEGEND_IMAGE, M.WALLS_LEGEND_PINCH,
                     M.WALLS_LEGEND_SPINES, M.WALLS_LEGEND_FLAT):
            text.encode("cp1252")
            assert text.strip().endswith(".")
            assert text[:1].isupper()
            assert "cell" not in text.lower(), "object, not cell"

    def test_a_refusal_can_take_the_legend_slot(self, widget):
        widget.set_walls_message("This patch is too large.")
        assert _legend_shown(widget) is True
        assert widget.split_walls_legend.text() == "This patch is too large."


class TestAGreyTickSaysWhy:
    """Qt shows NO tooltip on a disabled widget (fix pass H5).

    Every reason this control had for being grey was written into
    `setToolTip`, which on a disabled checkbox is never shown -- hover it
    and nothing happens.  So "why is that tick grey" had no answer at
    all.  An explicit reason is written into the LEGEND's slot as well,
    which is on screen exactly when S5 is, i.e. in review, which is the
    only state where a grey tick is a question worth asking.
    """

    def test_a_reason_is_on_screen_not_only_in_the_tooltip(self, widget):
        widget.set_walls_toggle_enabled(False, "Only a wall-based cut.")
        assert widget.split_walls_legend.text() == "Only a wall-based cut."
        assert _legend_shown(widget) is True

    def test_the_tooltip_still_carries_it_too(self, widget):
        widget.set_walls_toggle_enabled(False, "Only a wall-based cut.")
        assert widget.split_show_walls_cb.toolTip() == (
            "Only a wall-based cut.")

    def test_no_reason_means_an_empty_slot_not_a_default_sentence(
            self, widget):
        """"There is nothing worth saying" is a real answer, and it must
        not put the "Preview first" line on screen: a checkbox that is
        simply not applicable yet is explained by the whole card being
        absent."""
        M = _M()
        widget.set_walls_toggle_enabled(False)
        assert widget.split_walls_legend.text() == ""
        assert _legend_shown(widget) is False
        assert widget.split_show_walls_cb.toolTip() == M.WALLS_TOGGLE_DISABLED

    def test_enabling_hands_the_slot_back_to_the_legend(self, widget):
        widget.set_walls_toggle_enabled(False, "Only a wall-based cut.")
        widget.set_walls_toggle_enabled(True)
        assert widget.split_walls_legend.text() == ""
        assert _legend_shown(widget) is False

    def test_enabling_while_TICKED_puts_the_legend_back(self, widget):
        widget.set_walls_toggle_enabled(True)
        widget.split_show_walls_cb.setChecked(True)
        widget.set_walls_toggle_enabled(True)
        assert widget.split_walls_legend.text() == widget.walls_legend_text()
        assert _legend_shown(widget) is True

    def test_the_too_large_reason_points_at_the_line_below(self):
        M = _M()
        assert "below" in M.WALLS_TOGGLE_TOO_LARGE
        assert M.WALLS_TOGGLE_TOO_LARGE != M.WALLS_TOGGLE_DISABLED
        M.WALLS_TOGGLE_TOO_LARGE.encode("cp1252")

    def test_update_split_ui_state_carries_the_reason(self, widget):
        """The orchestrator's only door into the panel."""
        widget.update_split_ui_state(
            walls_toggle_enabled=False,
            walls_toggle_reason="Only a wall-based cut.")
        assert widget.split_walls_legend.text() == "Only a wall-based cut."

    def test_the_reason_keyword_defaults_to_None_like_every_other(self, widget):
        """`test_split_params_golden` freezes ONE rule over these setters:
        every keyword is keyword-only and defaults to ``None``.  H4 added
        this one with ``= ""`` and broke it -- the first default in the
        signature that was not None.

        The rule is not decoration: the setters are additive, and ``None``
        is how "the caller said nothing about this" is told apart from a
        value the caller meant.  Pinned here as well as in the golden file
        so the reason a green suite went red is written beside the
        feature, not only in the contract test.
        """
        import inspect

        p = inspect.signature(
            widget.update_split_ui_state).parameters["walls_toggle_reason"]
        assert p.default is None
        assert p.kind is p.KEYWORD_ONLY

    def test_no_reason_is_an_EMPTY_reason_not_the_string_None(self, widget):
        """...and the panel must still receive a `str`.

        ``set_walls_toggle_enabled`` writes its reason into a QLabel and
        into the tooltip; handed ``None`` it would render the word "None"
        at the user, which is worse than the wrong sentence H4 set out to
        fix.  Omitting the keyword has to behave exactly as ``""`` did.
        """
        widget.set_walls_toggle_enabled(False, "something earlier")
        widget.update_split_ui_state(walls_toggle_enabled=False)
        M = _M()
        assert widget.split_walls_legend.text() == ""
        assert _legend_shown(widget) is False
        assert widget.split_show_walls_cb.toolTip() == M.WALLS_TOGGLE_DISABLED

    def test_the_manual_reason_names_the_two_knobs_that_would_help(self):
        """H4.  A reason that only says "no" leaves the user stuck; this
        one names the marker picker and the wall picker, which are the
        only two things that turn this path into one with a picture."""
        M = _M()
        text = M.WALLS_TOGGLE_MANUAL
        assert "marker" in text.lower()
        assert "wall kind" in text.lower()
        assert M.WALLS_FROM_LABELS[M.WALLS_FROM_SHAPE] in text, (
            "it must name the wall kind the user is actually on")
        assert "Preview first" not in text
        text.encode("cp1252")


class TestTheHeightCeilingHolds:
    """UX round 1.4: review-with-markers-open had 6 px of its 753 px
    ceiling left, and 1.4 says that number may not be raised again
    without the owner.  The tick therefore has to cost the page NOTHING
    while it is OFF -- which is why it rides the RESULT header row."""

    def _shown(self, widget):
        from qtpy.QtWidgets import QApplication

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Reshape")
        QApplication.processEvents()
        return QApplication

    def _review_markers_open(self, widget):
        M = _M()
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        widget.update_split_ui_state(
            split_active=True, marker_ids=[1, 2, 3, 4, 5],
            marker_stats={i: 1000 * i for i in range(1, 6)})
        widget.update_split_guided_ui(
            pieces=[11, 12, 13],
            piece_sizes={11: 4000, 12: 3800, 13: 2000},
            caption="unticked pieces stay part of object 149",
            arbiter_message="some cuts had no wall behind them",
            split_anyway_visible=True,
            engine_messages=["u-Segment3D is not installed"])
        widget.update_split_ui_state(
            review_active=True, accept_enabled=True,
            refine_further_enabled=True, walls_toggle_enabled=True)

    def _measure(self, widget):
        slots = widget._panel_slots["Reshape"]
        screen = slots["scroll"].viewport().height()
        strip = getattr(widget, "_split_s4_container", None)
        if strip is not None and strip.parentWidget() is slots["page"]:
            screen += strip.height()
        return slots["content"].sizeHint().height(), screen

    def test_the_tick_costs_the_tallest_state_nothing_while_it_is_off(
            self, widget):
        """MEASURED 2026-09-21: 747 px, the same number Brick U4 left
        behind, against the 753 px ceiling."""
        app = self._shown(widget)
        try:
            self._review_markers_open(widget)
            app.processEvents()
            content, screen = self._measure(widget)
            assert content <= 3.1 * screen, (
                "review+markers is {} px in a {} px screen -- {:.2f} "
                "screens".format(content, screen, content / max(screen, 1))
            )
        finally:
            widget.hide()

    def test_ticking_it_costs_only_the_legend(self, widget):
        """UX round 1.4's rule, applied: the page may exceed the ceiling
        ONLY by the one line the user asked for.  MEASURED 2026-09-21:
        747 -> 775 px, i.e. the legend's 28 px and nothing else."""
        app = self._shown(widget)
        try:
            self._review_markers_open(widget)
            app.processEvents()
            off, _screen = self._measure(widget)
            widget.split_show_walls_cb.setChecked(True)
            app.processEvents()
            on, _screen = self._measure(widget)
            assert on > off, "the legend is on screen now"
            assert on - off <= 40, (
                "ticking it cost {} px -- the legend alone was measured "
                "at 28".format(on - off)
            )
        finally:
            widget.hide()


class TestTheLegacyPathSaysWhyItIsGrey:
    """H4.  Hand-painted markers plus "Pinch points (shape)" reach review
    without ever building an engine result, so there is no elevation and
    the tick stays grey.

    Nothing used to push a reason from that site, so the grey tick kept
    whatever the last state left on it -- "Preview first - there is
    nothing to show yet.", AFTER a Preview.
    """

    def _shown(self, widget):
        from qtpy.QtWidgets import QApplication

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Reshape")
        QApplication.processEvents()
        return QApplication

    def _legacy_review(self, widget):
        """What `_preview_split` pushes on the manual path."""
        M = _M()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_PAINTED)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_SHAPE)
        widget.update_split_ui_state(
            split_active=True, marker_ids=[1, 2],
            marker_stats={1: 900, 2: 800})
        widget.update_split_ui_state(
            review_active=True, accept_enabled=True,
            refine_further_enabled=True,
            walls_toggle_enabled=False,
            walls_toggle_reason=M.WALLS_TOGGLE_MANUAL)

    def test_the_reason_is_on_screen_after_a_preview(self, widget):
        M = _M()
        self._legacy_review(widget)
        assert widget.split_walls_legend.text() == M.WALLS_TOGGLE_MANUAL
        assert widget.split_show_walls_cb.isEnabled() is False

    def test_it_does_not_say_preview_first(self, widget):
        M = _M()
        self._legacy_review(widget)
        assert M.WALLS_TOGGLE_DISABLED not in (
            widget.split_walls_legend.text(),
            widget.split_show_walls_cb.toolTip(),
        )

    def test_the_reason_goes_when_the_preview_does(self, widget):
        M = _M()
        self._legacy_review(widget)
        widget.update_split_ui_state(review_active=False)
        assert widget.split_walls_legend.text() == ""

    def test_the_ceiling_still_holds_with_the_reason_on_screen(self, widget):
        """UX round 1.4: 753 px, and 1.4 says it may not be raised again
        without the owner.

        The reason is a line the user did NOT opt into -- unlike the
        legend, which is the deliberate exception 5 allows -- so it has
        to fit.  It does, comfortably: this path carries no amber block,
        no arbiter message and no engine messages, so its review state is
        far shorter than the guided one 1.4 measured.
        """
        app = self._shown(widget)
        try:
            widget.split_guided_summary_btn.setChecked(True)
            widget.split_markers_btn.setChecked(True)
            self._legacy_review(widget)
            app.processEvents()
            slots = widget._panel_slots["Reshape"]
            screen = slots["scroll"].viewport().height()
            strip = getattr(widget, "_split_s4_container", None)
            if strip is not None and strip.parentWidget() is slots["page"]:
                screen += strip.height()
            content = slots["content"].sizeHint().height()
            assert widget.split_walls_legend.isVisibleTo(
                widget._split_result_container), "measure it ON screen"
            assert content <= 3.1 * screen, (
                "legacy review with the reason up is {} px in a {} px "
                "screen -- {:.2f} screens".format(
                    content, screen, content / max(screen, 1))
            )
        finally:
            widget.hide()


class TestTheContractSurface:
    def test_the_methods_the_orchestrator_calls_exist(self, widget):
        for name in ("set_walls_toggle_enabled", "clear_walls_toggle",
                     "is_show_walls_checked", "set_walls_message",
                     "walls_legend_text"):
            assert callable(getattr(widget, name, None)), name

    def test_get_split_params_is_untouched(self, widget):
        """The tick is a DISPLAY control: it must never reach the
        engine."""
        before = widget.get_split_params()
        widget.set_walls_toggle_enabled(True)
        widget.split_show_walls_cb.setChecked(True)
        assert widget.get_split_params() == before
        assert "show_walls" not in before

    def test_is_show_walls_checked_answers_the_box(self, widget):
        assert widget.is_show_walls_checked() is False
        widget.set_walls_toggle_enabled(True)
        widget.split_show_walls_cb.setChecked(True)
        assert widget.is_show_walls_checked() is True


# --------------------------------------------------------------------------- #
# the orchestrator half
# --------------------------------------------------------------------------- #
class _Layer:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.visible = True
        self.editable = True
        self.mode = "pan_zoom"
        self.opacity = 1.0
        self.colormap = None

    def refresh(self):
        pass


class _Viewer:
    """Just a layer list.

    The production code builds the layer itself and APPENDS it (fix pass
    H2), so there is no `add_image` seam left to fake -- which is exactly
    the point: the flag exists before anything connected to the insert
    can read the list.
    """

    def __init__(self):
        self.layers = _LayerList()
        self.events = []


class _LayerList(list):
    """A layer list that fires an INSERT HOOK, synchronously.

    napari's ``layers.events.inserted`` does exactly that, and both
    filtered list builders are connected to it, so the moment a layer
    joins the list is the moment its flag has to be readable.
    """

    selection = type("S", (), {"active": None})()
    on_insert = None

    def append(self, layer):
        list.append(self, layer)
        if self.on_insert is not None:
            self.on_insert(layer)

    def remove(self, layer):
        list.remove(self, layer)


class _WallsAdv(FakeAdv):
    """FakeAdv plus the Brick U3 surface."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._walls_checked = False
        self._walls_enabled = False
        self.walls_messages = []
        self.walls_reasons = []
        self.walls_clears = 0

    def is_show_walls_checked(self):
        return bool(self._walls_checked)

    def set_walls_toggle_enabled(self, enabled, reason=""):
        self._walls_enabled = bool(enabled)
        self.walls_reasons.append(str(reason))

    def clear_walls_toggle(self):
        self._walls_checked = False
        self._walls_enabled = False
        self.walls_clears += 1

    def set_walls_toggle_checked(self, checked):
        self._walls_checked = bool(checked)

    def set_walls_message(self, text):
        self.walls_messages.append(str(text))

    def walls_legend_text(self):
        return "Bright = where the tool thinks an edge is."

    # the two panel channel combos `_on_layer_inserted` rebuilds
    def set_wall_channels(self, names, current=""):
        self.channels = list(names)

    def set_seed_channels(self, names, current=""):
        self.seed_channels = list(names)


def _stub_exit_seams(f):
    """Everything the six exits reach that a Qt-free fake has no answer
    for.  Deliberately no-ops rather than recorders: this file asserts
    what happened to the PICTURE, and the rest of each exit has its own
    suite."""
    for name in (
        "_abandon_reshape_worker", "_guide_thread_cleanup",
        "_remove_guide_layer", "_fire_reshape_lock",
        "_block_main_viewer_arrow_keys", "_invalidate_session_callbacks",
        "_reject_smooth", "_reject_add", "_reset_add_state",
        "_reject_interpolation", "_discard_batch_refine",
        "_teardown_batch_dialog", "_cancel_merge_dialog",
        "_reset_smooth_state", "_remove_prob_overlay",
        "_set_selected_label_all", "_sync_marker_parking",
        "_print_count_diag_recipe",
    ):
        setattr(f, name, lambda *a, **k: None)
    # The state a live session is actually in when the user hits an
    # exit: the object is punched out and a preview is on screen.  Both
    # matter -- `_leave_reshape_review` returns immediately without a
    # preview, and `_teardown_reshape_for_switch` takes the other side
    # of its gate without a running split.
    f._split_running = True
    f._split_previewed = True
    f._split_fragment_refine = False
    f._smooth_running = False
    f._load_locked = False
    f._interp_snapshot = None
    f._add_mode_running = False
    f._split_busy_kind = "solve"


def _spec_for(res):
    from packages.epicycle_annotator.reshape import ReshapeSpec

    return ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})


def _copy_result(res, **overrides):
    import copy

    out = copy.copy(res)
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def _result(elevation, partition=None):
    from packages.epicycle_annotator.reshape import ReshapeResult

    res = ReshapeResult()
    res.status = "ok"
    res.elevation = np.asarray(elevation, np.float32)
    res.result = (partition if partition is not None
                  else np.zeros_like(res.elevation, np.int32))
    return res


def _fake(shape=(6, 8, 8), roi=None, elev=None, main_scale=None,
          main_translate=None, units=None):
    """A fake annotator with a volume, a ROI and a solved result.

    The MAIN LABELS LAYER is a real object in the fake viewer, because
    the placement is read off it: `scale`, `translate` and `units` are
    what put the crop where the object is.
    """
    roi = roi or (slice(1, 4), slice(2, 6), slice(2, 6))
    volume = np.zeros(shape, np.int32)
    volume[roi] = 7
    adv = _WallsAdv()
    f = FakeAnnotator(volume, roi, 7, adv=adv,
                      refine=_Layer(np.zeros(shape, np.int32), "Refine"))
    f.viewer = _Viewer()
    main = _Layer(volume, "curation_3d_x")
    if main_scale is not None:
        main.scale = tuple(main_scale)
    if main_translate is not None:
        main.translate = tuple(main_translate)
    if units is not None:
        main.units = units
    f.viewer.layers.append(main)
    f._layer_by_name = lambda layers, name: next(
        (lyr for lyr in layers if getattr(lyr, "name", None) == name), None)
    crop_shape = tuple(sl.stop - sl.start for sl in roi)
    if elev is None:
        elev = np.arange(int(np.prod(crop_shape)), dtype=np.float32)
        elev = elev.reshape(crop_shape)
    f._reshape_result = _result(elev)
    f._split_previewed = True
    bind(f, "_on_reshape_show_walls_toggled", "_show_walls_picture",
         "_drop_walls_picture", "_walls_display_crop")
    return f, adv


@pytest.fixture(scope="module")
def solved():
    """A genuine two-piece ``ReshapeResult``, elevation and all.

    The source-reading tests this file used to carry (L1) are driven
    against this instead: `_apply_reshape_result` takes a result, so
    give it one.
    """
    from packages.epicycle_annotator.reshape import ReshapeSpec, reshape_region

    from .reshape_fixtures import two_columns

    labels, guide, membrane, _ = two_columns(0.30)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    res = reshape_region(labels, {"membrane": membrane}, spec,
                         target_id=7, seed_layer=guide)
    assert res.status == "ok" and res.elevation is not None
    return labels, res


def _applied_fake(solved):
    """A fake sitting where a finished solve hands its result over."""
    labels, res = solved
    volume = labels.copy()
    adv = _WallsAdv()
    f = FakeAnnotator(volume, res.roi, 7, adv=adv,
                      refine=_Layer(np.zeros_like(labels), "Refine"))
    f.viewer = _Viewer()
    f.viewer.layers.append(_Layer(volume, "curation_3d_x"))
    f._layer_by_name = lambda layers, name: next(
        (lyr for lyr in layers if getattr(lyr, "name", None) == name), None)
    bind(f, "_apply_reshape_result", "_print_count_diag_result",
         "_render_reshape_preview", "_reshape_quality_notes",
         "_reshape_review_status", "_on_reshape_show_walls_toggled")
    return f, adv, res


class TestTheWallsCrop:
    """The layer is the ROI CROP placed with ``translate`` + ``scale``,
    not a volume-shaped array.

    asked for a full-size array to get ALIGNMENT.  The live driver
    then measured what that costs on the owner's own patch --
    (317, 480, 480) is 292 MB, refused by the byte budget on every
    Preview of the run -- and `translate` gives the same alignment for
    the crop's 2.7 MB.  These pin the arithmetic that makes that safe,
    because a wall picture in the WRONG PLACE is the worst failure this
    control has.
    """

    def test_the_layer_data_IS_the_elevation(self):
        f, _adv = _fake()
        crop, _t, _sc, _u = f._walls_display_crop()
        assert crop.shape == f._reshape_result.elevation.shape
        assert crop.dtype == np.float32
        np.testing.assert_array_equal(crop, f._reshape_result.elevation)

    def test_it_allocates_nothing_volume_shaped(self):
        """The whole point: the crop is 1/110th of the volume on the
        owner's patch, and nothing volume-shaped is built at all."""
        f, _adv = _fake()
        crop, _t, _sc, _u = f._walls_display_crop()
        volume = f.sync_manager.shared_labels["curation_3d_x"]
        assert crop.size < volume.size
        assert crop.shape == tuple(
            sl.stop - sl.start for sl in f._split_roi)

    def test_the_translate_is_the_roi_start(self):
        f, _adv = _fake()
        _crop, translate, scale, _u = f._walls_display_crop()
        assert translate == (1.0, 2.0, 2.0)
        assert scale == (1.0, 1.0, 1.0)

    def test_a_scaled_main_layer_scales_AND_offsets_the_crop(self):
        """napari composites in world coordinates:
        ``world = translate + scale x data``.  A crop whose index ``d``
        is volume index ``d + roi_start`` therefore needs
        ``translate = main.translate + main.scale x roi_start`` -- the
        offset has to be scaled too, or the picture drifts further from
        the object the further into the volume it is."""
        f, _adv = _fake(main_scale=(2.0, 0.5, 0.5))
        _crop, translate, scale, _u = f._walls_display_crop()
        assert scale == (2.0, 0.5, 0.5)
        assert translate == (2.0, 1.0, 1.0)

    def test_a_translated_main_layer_is_added_on_top(self):
        f, _adv = _fake(main_scale=(2.0, 1.0, 1.0),
                        main_translate=(10.0, 0.0, -3.0))
        _crop, translate, _sc, _u = f._walls_display_crop()
        assert translate == (12.0, 2.0, -1.0)

    def test_the_units_follow_the_main_layer(self):
        f, _adv = _fake(units=("um", "um", "um"))
        _crop, _t, _sc, units = f._walls_display_crop()
        assert units == ("um", "um", "um")

    def test_it_answers_with_FOUR_values(self):
        """The docstring said three while every caller unpacked four
        (M5).  Pinned, so the two cannot disagree again."""
        f, _adv = _fake()
        assert len(f._walls_display_crop()) == 4

    def test_no_result_means_no_crop(self):
        f, _adv = _fake()
        f._reshape_result = None
        assert f._walls_display_crop() is None

    def test_a_result_with_no_elevation_means_no_crop(self):
        f, _adv = _fake()
        f._reshape_result.elevation = None
        assert f._walls_display_crop() is None

    def test_a_shape_mismatch_is_refused_rather_than_placed(self):
        """A crop that does not match the ROI belongs to another
        session.  Half a wall picture in the wrong place would read as
        "the tool saw THAT"."""
        f, _adv = _fake()
        f._reshape_result.elevation = np.zeros((2, 2, 2), np.float32)
        assert f._walls_display_crop() is None

    def test_a_wrong_dimensionality_is_refused(self):
        f, _adv = _fake()
        f._reshape_result.elevation = np.zeros((3, 4), np.float32)
        assert f._walls_display_crop() is None

    def test_no_main_layer_is_a_REFUSAL_not_an_identity_placement(self):
        """H3 (LOW).  The placement is read off the main labels layer.

        When that layer cannot be found, the old code fell through to
        identity -- scale 1, translate 0 -- and placed the crop at the
        bare ROI start.  That is only right if the object's own layer
        happens to be unscaled and untranslated, and there is no way to
        know: under a physical voxel size it is the wrong place, quietly.
        A picture in the wrong place is worse than none.
        """
        f, _adv = _fake(main_scale=(2.0, 0.5, 0.5))
        f._layer_by_name = lambda _layers, _name: None
        assert f._walls_display_crop() is None

    def test_the_missing_main_layer_says_so(self, capsys):
        f, _adv = _fake()
        f._layer_by_name = lambda _layers, _name: None
        f._on_reshape_show_walls_toggled(True)
        out = capsys.readouterr().out
        assert "[WALLS]" in out and "refused" in out
        assert "walls" not in f._temp_guide_layers


class TestTicking:
    def test_it_adds_one_temporary_image_layer_called_guide_walls(self):
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        lyr = f._temp_guide_layers.get("walls")
        assert lyr is not None
        assert lyr.name == "guide_walls"
        assert lyr._epicycle_temporary is True
        assert lyr.editable is False
        assert lyr.mode == "pan_zoom"

    def test_it_is_magma_at_half_opacity(self):
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        lyr = f._temp_guide_layers["walls"]
        assert lyr.colormap.name == "magma"
        assert lyr.opacity == 0.5

    def test_it_is_a_real_image_layer(self):
        """Not a stand-in: the production code constructs the layer, so
        everything downstream -- `isinstance(layer, Image)` in every list
        builder, the `_epicycle_temporary` skip, the vispy visual -- sees
        the same object the user does."""
        from napari.layers import Image as _Image

        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        assert isinstance(f._temp_guide_layers["walls"], _Image)

    def test_the_layer_carries_the_elevation_AND_its_placement(self):
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        lyr = f._temp_guide_layers["walls"]
        np.testing.assert_array_equal(lyr.data, f._reshape_result.elevation)
        assert tuple(lyr.translate) == (1.0, 2.0, 2.0)
        assert tuple(lyr.scale) == (1.0, 1.0, 1.0)

    def test_the_placement_reaches_the_layer_under_a_scale(self):
        f, _adv = _fake(main_scale=(2.0, 0.5, 0.5),
                        main_translate=(10.0, 0.0, -3.0))
        f._on_reshape_show_walls_toggled(True)
        lyr = f._temp_guide_layers["walls"]
        assert tuple(lyr.scale) == (2.0, 0.5, 0.5)
        assert tuple(lyr.translate) == (12.0, 1.0, -2.0)

    def test_the_crop_index_maps_back_onto_the_object(self):
        """The check that would catch a placement bug the way the user
        would: walk the crop's own indices through `translate` and land
        on the object's voxels in the volume."""
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        lyr = f._temp_guide_layers["walls"]
        volume = f.sync_manager.shared_labels["curation_3d_x"]
        offset = np.asarray(lyr.translate, int)
        coords = np.argwhere(np.asarray(lyr.data) >= 0)
        landed = volume[tuple((coords + offset).T)]
        assert set(np.unique(landed)) == {7}, (
            "every voxel of the picture must land on the object")

    def test_the_main_labels_layer_and_the_refine_layer_are_untouched(self):
        f, _adv = _fake()
        volume = f.sync_manager.shared_labels["curation_3d_x"]
        before_vol = volume.copy()
        before_refine = f._refine_layer.data.copy()
        f._on_reshape_show_walls_toggled(True)
        np.testing.assert_array_equal(volume, before_vol)
        np.testing.assert_array_equal(f._refine_layer.data, before_refine)

    def test_ticking_with_no_result_adds_nothing(self):
        f, adv = _fake()
        f._reshape_result = None
        f._on_reshape_show_walls_toggled(True)
        assert "walls" not in f._temp_guide_layers
        assert adv.walls_clears >= 1, "the box goes back down"


class TestThePlacementFallback:
    """A layer in the right place with no units is fine; a layer that
    never appeared is not.  So the placement kwargs are dropped on ANY
    failure, not only a TypeError -- a napari that TAKES `units` but
    rejects the value would otherwise kill the picture outright."""

    def _picky(self, monkeypatch, *bad_keys):
        """Refuse named kwargs at the CONSTRUCTOR -- where the production
        code now builds the layer (fix pass H2)."""
        K = _mod()
        real = K.Image

        def _make(data, **kw):
            for key in bad_keys:
                if key in kw:
                    raise ValueError(
                        "this napari dislikes {!r}".format(key))
            return real(data, **kw)

        monkeypatch.setattr(K, "Image", _make)

    def test_units_are_dropped_rather_than_losing_the_layer(
            self, monkeypatch):
        f, _adv = _fake(units=("um", "um", "um"))
        self._picky(monkeypatch, "units")
        f._on_reshape_show_walls_toggled(True)
        lyr = f._temp_guide_layers.get("walls")
        assert lyr is not None, "the picture must still appear"
        assert tuple(lyr.translate) == (1.0, 2.0, 2.0), (
            "and it must still be in the right place")

    def test_an_unplaceable_picture_is_REFUSED_not_misplaced(
            self, monkeypatch):
        """H3.  The third attempt used to be the BARE call, which added
        the crop at the volume ORIGIN.

        That is not a degraded wall picture, it is a WRONG one: a
        greyscale block sitting off the object, captioned "this is what
        the tool saw".  With a placement to honour, failing to honour it
        is a refusal.
        """
        f, adv = _fake()
        self._picky(monkeypatch, "translate")
        f._on_reshape_show_walls_toggled(True)
        assert "walls" not in f._temp_guide_layers
        assert not [l for l in f.viewer.layers
                    if getattr(l, "name", "") == "guide_walls"], (
            "nothing unplaced may be left in the viewer")
        assert adv.walls_clears >= 1, "and the box goes back down"
        assert adv.walls_messages, "and the user is told, not left guessing"

    def test_a_scale_refusal_is_a_refusal_too(self, monkeypatch):
        f, _adv = _fake(main_scale=(2.0, 0.5, 0.5))
        self._picky(monkeypatch, "scale")
        f._on_reshape_show_walls_toggled(True)
        assert "walls" not in f._temp_guide_layers

    def test_the_refusal_says_so_on_the_console(self, monkeypatch, capsys):
        f, _adv = _fake()
        self._picky(monkeypatch, "translate")
        f._on_reshape_show_walls_toggled(True)
        out = capsys.readouterr().out
        assert "[WALLS]" in out
        for line in out.splitlines():
            line.encode("cp1252")

    def test_a_layer_with_NO_placement_still_has_its_bare_call(
            self, monkeypatch):
        """The rule is about a placement that exists and was refused.
        The guide MASK carries none, so its one attempt IS the bare call
        and nothing about it changed."""
        f, _adv = _fake()
        lyr = f._add_temp_layer("mask", np.zeros((3, 4, 4), np.int32),
                                kind="labels", name="guide_mask")
        assert lyr is not None
        assert lyr in list(f.viewer.layers)

    def test_a_real_failure_still_reports_None(self, monkeypatch):
        f, adv = _fake()
        self._picky(monkeypatch, "name")     # every attempt fails
        f._on_reshape_show_walls_toggled(True)
        assert "walls" not in f._temp_guide_layers
        assert adv.walls_clears >= 1, "and the box goes back down"


class TestUnticking:
    def test_it_removes_the_layer(self):
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        f._on_reshape_show_walls_toggled(False)
        assert "walls" not in f._temp_guide_layers
        assert not [l for l in f.viewer.layers if l.name == "guide_walls"]

    def test_it_hides_before_it_removes(self):
        """The vispy visual has to leave the scene graph
        before the GPU texture is freed."""
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        lyr = f._temp_guide_layers["walls"]
        f._on_reshape_show_walls_toggled(False)
        assert lyr.visible is False

    def test_it_is_idempotent(self):
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(False)
        f._on_reshape_show_walls_toggled(False)
        assert "walls" not in f._temp_guide_layers


class TestTheFlagBeatsTheInsertListeners:
    """``_epicycle_temporary`` has to exist BEFORE the layer is inserted.

    ``layers.events.inserted`` fires synchronously from the add, and the
    two filtered list builders are connected to it -- the annotator's own
    ``_refresh_layer_options`` and, on the Step-3 subclass,
    ``_refresh_all_image_combos``, which fills the five panel channel
    combos.  Setting the flag on the RETURNED layer is one statement too
    late: by then every one of them has walked a list holding an
    unflagged ``guide_walls`` and offered it as a channel, for the whole
    life of the picture.  Picking it there feeds the wall ELEVATION back
    into the engine as if it were an image channel.

    So the layer is built, flagged, and only then appended.
    """

    def test_the_layer_is_flagged_before_it_reaches_the_layer_list(self):
        seen = []
        f, _adv = _fake()
        f.viewer.layers.on_insert = lambda lyr: seen.append(
            bool(getattr(lyr, "_epicycle_temporary", False)))
        f._on_reshape_show_walls_toggled(True)
        assert seen == [True], (
            "the insert listeners saw an unflagged layer: {}".format(seen))

    def test_the_guide_mask_is_flagged_first_as_well(self):
        """The registry's other user.  A Labels guide is kept out of the
        LABELS combo by the same flag, on the same event."""
        seen = []
        f, _adv = _fake()
        f.viewer.layers.on_insert = lambda lyr: seen.append(
            bool(getattr(lyr, "_epicycle_temporary", False)))
        f._add_temp_layer("mask", np.zeros((3, 4, 4), np.int32),
                          kind="labels", name="guide_mask")
        assert seen == [True]

    def test_no_channel_combo_lists_it(self):
        """End to end, through the REAL builders.

        ``_refresh_wall_channels`` and ``_refresh_seed_channels`` are the
        production methods behind two of the five channel combos, and
        both already skip ``_epicycle_temporary``.  Run them from the
        insert hook -- which is where ``_on_layer_inserted`` runs them --
        and the picture must not be in either list.
        """
        f, adv = _fake()
        bind(f, "_refresh_wall_channels", "_refresh_seed_channels")
        adv.channels = adv.seed_channels = None

        def _inserted(_lyr):
            f._refresh_wall_channels()
            f._refresh_seed_channels()

        f.viewer.layers.on_insert = _inserted
        f._on_reshape_show_walls_toggled(True)
        assert adv.channels == [], (
            "the wall-channel combo offered {}".format(adv.channels))
        assert adv.seed_channels == []

    def test_a_real_channel_is_still_offered(self):
        """The contrast, so the assertion above cannot pass by the
        builders never running."""
        from napari.layers import Image as _Image

        f, adv = _fake()
        bind(f, "_refresh_wall_channels")
        f.viewer.layers.append(_Image(np.zeros((3, 4, 4), np.float32),
                                      name="membrane"))
        adv.channels = None
        f.viewer.layers.on_insert = lambda _l: f._refresh_wall_channels()
        f._on_reshape_show_walls_toggled(True)
        assert adv.channels == ["membrane"]


class TestTheByteBudget:
    """64 MB, and it is a guard on the pathological case again.

    The number was right and the IMPLEMENTATION was wrong: the
    first build put the elevation into a volume-shaped array, so the
    budget was being asked about the whole patch rather than about the
    picture.  On the owner's own data that is 292 MB and the tick was
    refused on every Preview of the live driver run.  With the layer as
    the CROP the same object needs 2.7 MB, and 64 MB is what it always
    meant: a ceiling a real curation patch does not walk into.
    """

    #: MEASURED, live driver 2026-09-21, object 596 on the dev workspace.
    REAL_VOLUME = (317, 480, 480)
    REAL_CROP = (69, 99, 97)            # roi ((34,103),(45,144),(59,156))

    def _budget(self):
        K = __import__(
            "napari_epicycle.annotator.curation_3d_widget",
            fromlist=["x"])
        return K.WALLS_DISPLAY_BUDGET_BYTES

    def test_the_constant_is_a_byte_count_of_its_own(self):
        """NOT `spec.DEFAULT_SIZE_BUDGET`, which is a voxels x seeds
        PRODUCT -- reusing it would be a unit error hiding as a
        constant."""
        assert self._budget() == 64 * 1024 * 1024

    def test_the_owners_real_CROP_is_admitted(self):
        """69x99x97 float32 = 2.7 MB, a twenty-fourth of the budget."""
        crop_bytes = int(np.prod(self.REAL_CROP)) * 4
        assert crop_bytes == 2_650_428
        assert crop_bytes < self._budget()

    def test_the_owners_real_VOLUME_would_be_refused(self):
        """The regression in one line.  Had the layer stayed
        volume-shaped, this is the number the budget would be asked
        about -- 292 MB, refused, on every Preview of the run.  If a
        future change goes back to a full-size array, this test says
        what it costs."""
        volume_bytes = int(np.prod(self.REAL_VOLUME)) * 4
        assert volume_bytes == 292_147_200
        assert volume_bytes > self._budget()

    def test_the_budget_is_measured_against_the_CROP(self):
        """Runtime, not source-reading: a fake whose crop is tiny but
        whose VOLUME is enormous must still be shown."""
        f, adv = _fake(shape=(400, 400, 400),
                       roi=(slice(10, 14), slice(20, 24), slice(30, 34)))
        f._on_reshape_show_walls_toggled(True)
        assert "walls" in f._temp_guide_layers, (
            "a 4x4x4 picture in a 400-cube volume must not be refused")
        assert not adv.walls_messages

    def test_a_crop_over_budget_is_refused_in_plain_words(self):
        f, adv = _fake()
        K = __import__(
            "napari_epicycle.annotator.curation_3d_widget",
            fromlist=["x"])
        old = K.WALLS_DISPLAY_BUDGET_BYTES
        try:
            K.WALLS_DISPLAY_BUDGET_BYTES = 8       # bytes: two voxels
            f._on_reshape_show_walls_toggled(True)
        finally:
            K.WALLS_DISPLAY_BUDGET_BYTES = old
        assert "walls" not in f._temp_guide_layers
        assert adv.walls_messages, "the user is told why"
        msg = adv.walls_messages[-1]
        assert "too large" in msg.lower()
        # ...and it names the REGION, not the patch (fix pass M2).  The
        # layer is the object's crop, so the budget is measured against
        # the region; a user told the PATCH is too big would go looking
        # for a smaller patch rather than a smaller object.
        assert "region" in msg.lower()
        assert "patch" not in msg.lower()
        msg.encode("cp1252")
        assert adv.walls_clears >= 1, "and the box goes back down"

    def test_the_grey_tick_stops_saying_preview_first(self):
        """H5.  A Preview is exactly what just happened, so the default
        reason is the one sentence that cannot be true here."""
        from napari_epicycle.annotator.advanced_features_widget import (
            WALLS_TOGGLE_DISABLED, WALLS_TOGGLE_TOO_LARGE,
        )

        f, adv = _fake()
        K = _mod()
        old = K.WALLS_DISPLAY_BUDGET_BYTES
        try:
            K.WALLS_DISPLAY_BUDGET_BYTES = 8
            f._on_reshape_show_walls_toggled(True)
        finally:
            K.WALLS_DISPLAY_BUDGET_BYTES = old
        assert WALLS_TOGGLE_TOO_LARGE in adv.walls_reasons
        assert adv.walls_reasons[-1] != WALLS_TOGGLE_DISABLED

    def test_64_MB_of_float32_is_16_million_voxels(self):
        """What the cap means in the units the user would think in."""
        assert self._budget() // 4 == 16 * 1024 * 1024


class TestTheRebuildOnARePreview:
    """A re-Preview must REPLACE the picture, never leave the previous
    cut's on screen next to the new pieces (audit M2)."""

    def test_the_intent_survives_the_preview_that_unticks_the_box(self):
        """`_walls_wanted`, not the checkbox.  Pressing "Preview again"
        pushes `review_active=False` through the panel, which returns
        every result-shaped control to idle -- the tick with them.  A
        rebuild keyed on the BOX would therefore be dead code."""
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        assert f._walls_wanted is True
        # what `_preview_reshape` does while the solve runs
        f._remove_temp_layer("walls")
        assert f._walls_wanted is True, (
            "the stale picture comes off; the intent does not")

    def test_the_new_elevation_replaces_the_old_one(self):
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        first = f._temp_guide_layers["walls"]

        crop = tuple(sl.stop - sl.start for sl in f._split_roi)
        f._reshape_result = _result(np.full(crop, 9.0, np.float32))
        assert f._show_walls_picture() is True
        second = f._temp_guide_layers["walls"]
        assert second is not first, "the picture is rebuilt, not reused"
        assert float(second.data[f._split_roi].max()) == 9.0
        assert len([l for l in f.viewer.layers
                    if l.name == "guide_walls"]) == 1

    def test_the_box_is_put_back_up_to_agree_with_the_viewer(self):
        f, adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        adv._walls_checked = False          # what review_active=False did
        f._show_walls_picture()
        if f._walls_wanted:
            adv.set_walls_toggle_checked(True)
        assert adv.is_show_walls_checked() is True

    def test_the_result_rebuilds_the_picture_when_it_is_wanted(self, solved):
        """DRIVEN (L1), not read off the source: hand
        `_apply_reshape_result` a real engine result with the intent
        held, and the picture must be on screen when it returns."""
        f, adv, res = _applied_fake(solved)
        f._walls_wanted = True
        f._apply_reshape_result(res)
        assert "walls" in f._temp_guide_layers
        assert adv.is_show_walls_checked() is True, (
            "the box has to agree with the viewer again")

    def _pushed(self, f, key):
        for call in reversed(f.ui_calls):
            if key in call:
                return call[key]
        return None

    def test_the_result_offers_the_tick_even_when_it_is_not_wanted(
            self, solved):
        f, _adv, res = _applied_fake(solved)
        f._walls_wanted = False
        f._apply_reshape_result(res)
        assert self._pushed(f, "walls_toggle_enabled") is True, (
            "there IS an elevation now")
        assert "walls" not in f._temp_guide_layers, (
            "...but nothing was asked for, so nothing is added")

    def test_a_result_with_no_elevation_leaves_the_tick_grey(self, solved):
        """The manual path's twin, on the engine side: no elevation, no
        picture to offer."""
        f, _adv, res = _applied_fake(solved)
        f._walls_wanted = False
        f._apply_reshape_result(_copy_result(res, elevation=None))
        assert self._pushed(f, "walls_toggle_enabled") is False

    def test_preview_takes_the_stale_picture_off(self, solved):
        """DRIVEN (L1).  `_preview_reshape` cannot be run to completion
        on a Qt-free fake -- it ends in a real `QThread` -- but the
        removal happens well before that, so the launch is stopped at
        `_lock_subviewers_for_preview`, which sits OUTSIDE the method's
        own try/except and so propagates.

        What it must do: take the LAYER off, because the picture on
        screen belongs to the cut being replaced, and KEEP the intent,
        because a new solve is not an exit and the result has to be able
        to rebuild it.
        """
        f, adv, res = _applied_fake(solved)
        f._walls_wanted = True
        f._apply_reshape_result(res)
        assert "walls" in f._temp_guide_layers

        class _Stop(BaseException):
            """Not an `Exception`, so the launch guard cannot swallow
            it."""

        bind(f, "_preview_reshape")
        f._reshape_recipe = lambda: (
            _spec_for(res), {"membrane": np.zeros(f._split_snapshot.shape,
                                                  np.float32)}, None)
        f._reshape_labels_for_engine = lambda: (
            f.sync_manager.shared_labels["curation_3d_x"])
        f._reshape_promoted = False
        f._print_count_diag_recipe = lambda *a, **k: None

        def _stop():
            raise _Stop()

        f._lock_subviewers_for_preview = _stop
        with pytest.raises(_Stop):
            f._preview_reshape()

        assert "walls" not in f._temp_guide_layers, (
            "the stale picture must come off while the new solve runs")
        assert f._walls_wanted is True, (
            "a new solve is not an EXIT -- dropping the intent would mean "
            "the rebuild never fires")


class TestTheLegacyPreviewPushesTheReason:
    """H4, driven rather than read.

    A hand-painted two-marker preview against "Pinch points (shape)"
    never reaches the engine, so `_apply_reshape_result` -- the only site
    that used to push `walls_toggle_enabled` -- never runs.  The grey
    tick therefore kept whatever the previous state left on it, which is
    the default "Preview first - there is nothing to show yet.", AFTER a
    Preview.
    """

    def _legacy(self):
        from .test_reshape_split_orchestration import FakeSync

        shape = (6, 10, 10)
        labels = np.zeros(shape, np.int32)
        labels[1:5, 1:9, 1:9] = 7
        roi = (slice(0, 6), slice(0, 10), slice(0, 10))
        vol = labels.copy()
        vol[vol == 7] = 0
        refine = _Layer(labels.copy(), "Refine")
        refine.data[2, 3, 3] = 101
        refine.data[2, 6, 6] = 102
        adv = _WallsAdv()
        f = FakeAnnotator(vol, roi, 7, adv=adv, refine=refine)
        f.sync_manager = FakeSync({"curation_3d_x": vol})
        f._split_snapshot = labels[roi].copy()
        f._refine_baseline = labels[roi].copy()
        f._split_painted_marker_ids = [101, 102]
        f._reshape_active = False
        f.viewer = _Viewer()
        bind(f, "_preview_split", "_marker_rows_from_refine",
             "_apply_marker_unticks", "_painted_marker_ids_on_layer",
             "_checked_marker_ids")
        f._arm_guided_if_needed = lambda: False
        f._is_guided_now = lambda: False
        return f, adv

    def _last(self, f, key):
        for call in reversed(f.ui_calls):
            if key in call:
                return call[key]
        return None

    def test_it_pushes_the_manual_reason(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            WALLS_TOGGLE_MANUAL,
        )

        f, _adv = self._legacy()
        f._preview_split()
        assert f._split_previewed is True, f.ui_calls[-1:]
        assert self._last(f, "walls_toggle_enabled") is False
        assert self._last(f, "walls_toggle_reason") == WALLS_TOGGLE_MANUAL

    def test_it_reaches_review_at_all(self):
        """The guard on the two assertions above: they would both pass
        vacuously if this path stopped reaching review."""
        f, _adv = self._legacy()
        f._preview_split()
        assert self._last(f, "review_active") is True


class TestEveryExitTakesItAway:
    """Six sites: the five temp-layer ones plus the
    snap-back, which is the one the loop reaches most often."""

    EXITS = (
        "_leave_reshape_review",
        "_accept_split",
        "_reject_split",
        "_on_reshape_cancel",
        "_teardown_reshape_for_switch",
        "_close_cleanup",
    )

    @pytest.mark.parametrize("name", EXITS)
    def test_the_exit_drops_the_picture(self, name):
        """DRIVEN (L1), not read off the source.

        Each exit is CALLED with the picture up.  Most of them go on to
        do things a Qt-free fake cannot follow -- rebuild subviewers,
        talk to a navigator, close a dock -- so the call is allowed to
        raise AFTERWARDS; what is asserted is the state it leaves the
        picture in, which is the thing that matters and the thing a
        string match could never prove.  A drop that moved to the end of
        one of these methods would fail here, which is the point: the
        old assertion would have passed.
        """
        f, adv = _fake()
        _stub_exit_seams(f)
        # `_close_cleanup` reaches its own drop only after calling
        # `_reject_split`, which has one of its own -- so the real
        # one is bound rather than stubbed, or the test would prove
        # the wrong half.
        bind(f, name, "_restore_reshape_target", "_reject_split")
        f._on_reshape_show_walls_toggled(True)
        adv._walls_checked = True
        assert "walls" in f._temp_guide_layers, "the fixture must show it"

        try:
            if name == "_leave_reshape_review":
                getattr(f, name)("Markers changed - press Preview again.")
            else:
                getattr(f, name)()
        except BaseException as exc:            # noqa: BLE001
            print("{} stopped at {!r} -- after the drop".format(name, exc))

        assert "walls" not in f._temp_guide_layers, (
            "{} leaves the wall picture on screen".format(name))
        assert not [l for l in f.viewer.layers
                    if getattr(l, "name", "") == "guide_walls"]
        assert f._walls_wanted is False, (
            "{} is an EXIT: it forgets the intent too".format(name))

    def test_drop_unticks_the_box_as_well_as_removing_the_layer(self):
        f, adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        adv._walls_checked = True
        f._drop_walls_picture()
        assert "walls" not in f._temp_guide_layers
        assert adv._walls_checked is False
        assert adv.walls_clears >= 1

    def test_drop_is_idempotent_and_never_raises_without_a_panel(self):
        f, _adv = _fake()
        f._adv = None
        f._drop_walls_picture()
        f._drop_walls_picture()

    def test_the_snap_back_drops_it(self):
        """The loop's own exit: any change in S3 puts the preview away,
        and the wall picture is part of the preview."""
        f, adv = _fake()
        bind(f, "_leave_reshape_review")
        f._on_reshape_show_walls_toggled(True)
        adv._walls_checked = True
        f._split_previewed = True
        f._leave_reshape_review("Markers changed - press Preview to cut "
                                "again.")
        assert "walls" not in f._temp_guide_layers
        assert adv._walls_checked is False

    def test_a_patch_switch_sweeps_it_with_the_rest(self):
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        assert "walls" in f._temp_guide_layers
        f._remove_all_temp_layers()
        assert "walls" not in f._temp_guide_layers

    def test_the_sweep_alone_does_NOT_clear_the_intent(self):
        """Which is why the teardown needs the dropper as well (M5).

        `_remove_all_temp_layers` knows about LAYERS.  `_walls_wanted` is
        an orchestrator flag it has never heard of.
        """
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        f._remove_all_temp_layers()
        assert f._walls_wanted is True

    def test_the_teardown_clears_the_intent_too(self):
        """M5.  Without this the flag rode into the NEXT patch, where the
        first Preview would rebuild a wall picture nobody asked for --
        `_apply_reshape_result` reads the intent, not the box."""
        f, adv = _fake()
        bind(f, "_teardown_reshape_for_switch")
        f._abandon_reshape_worker = lambda: None
        f._guide_thread_cleanup = lambda: None
        f._remove_guide_layer = lambda: None
        f._fire_reshape_lock = lambda _on: None
        f._split_running = False          # nothing punched out: the gate
        f._on_reshape_show_walls_toggled(True)
        adv._walls_checked = True

        f._teardown_reshape_for_switch()

        assert "walls" not in f._temp_guide_layers
        assert f._walls_wanted is False
        assert adv._walls_checked is False

    def test_the_teardown_clears_it_with_the_object_punched_out_too(self):
        """The same exit on the other side of the `_split_running` gate:
        it must not matter which, because a user switching patch mid-cut
        is the commoner case."""
        f, adv = _fake()
        bind(f, "_teardown_reshape_for_switch", "_restore_reshape_target")
        f._abandon_reshape_worker = lambda: None
        f._guide_thread_cleanup = lambda: None
        f._remove_guide_layer = lambda: None
        f._fire_reshape_lock = lambda _on: None
        f._split_running = True
        f._on_reshape_show_walls_toggled(True)

        f._teardown_reshape_for_switch()

        assert "walls" not in f._temp_guide_layers
        assert f._walls_wanted is False


class TestTheDiagnostics:
    @pytest.mark.parametrize("phrase", ("[WALLS]",))
    def test_the_prefix_is_used(self, phrase, capsys):
        f, _adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        f._on_reshape_show_walls_toggled(False)
        out = capsys.readouterr().out
        assert phrase in out

    def test_every_walls_line_is_cp1252_safe(self, capsys):
        f, adv = _fake()
        f._on_reshape_show_walls_toggled(True)
        f._show_walls_picture()
        f._drop_walls_picture()
        f._reshape_result = None
        f._on_reshape_show_walls_toggled(True)
        for line in capsys.readouterr().out.splitlines():
            line.encode("cp1252")

    def test_the_refusal_prints_the_byte_count(self, capsys):
        f, _adv = _fake()
        K = __import__(
            "napari_epicycle.annotator.curation_3d_widget",
            fromlist=["x"])
        old = K.WALLS_DISPLAY_BUDGET_BYTES
        try:
            K.WALLS_DISPLAY_BUDGET_BYTES = 8
            f._on_reshape_show_walls_toggled(True)
        finally:
            K.WALLS_DISPLAY_BUDGET_BYTES = old
        out = capsys.readouterr().out
        assert "refused" in out and "bytes" in out

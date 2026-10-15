"""`_apply_themed_styles` must still REACH every input after the re-layout.

It walks ``self.findChildren(...)`` once at build time, so a widget that
moves into a container which is not a descendant of the panel silently
loses its theme -- and there is no visual test in this suite.  The panel
rework re-parents almost every control into scroll areas and slot
containers, which is exactly the move that would break it.

The second half of the file pins the Brick 2 MOCKUP's visual language

onto the widgets that must carry it: the card, its UPPERCASE header, the
one fixed label column per section, the chip, the ticklist row, the grey
helper line, the amber block, the status pill, the gold accent and the
footer.  Every assertion is made against the LIVE widget (a stylesheet a
widget actually carries, a width it actually has), never against the
source -- a source grep passes happily while the style is set on a
widget nobody can see.

Pattern 3 (offscreen QApplication) .
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")


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


def _inputs(widget):
    from qtpy.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QLineEdit,
        QSlider,
        QSpinBox,
    )

    out = []
    for cls in (QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QSlider):
        out.extend(widget.findChildren(cls))
    return out


class TestThemedStyleReach:
    def test_every_input_carries_a_stylesheet(self, widget):
        bare = [w for w in _inputs(widget) if not w.styleSheet()]
        assert not bare, (
            "{} input(s) were missed by _apply_themed_styles -- they are no "
            "longer children of the panel when the sweep runs: {}".format(
                len(bare), [type(w).__name__ for w in bare][:10])
        )

    def test_the_contract_inputs_are_all_reached(self, widget):
        """Spot-check the named knobs the orchestrator reads, per page."""
        for name in (
            "split_solver_combo", "split_rw_beta_sb", "split_min_size_sb",
            "split_wall_channel_combo", "split_detect_min_dist_sb",
            "add_channel_combo", "add_min_size_sb", "add_preset_combo",
            "add_marker_source_combo", "add_wall_channel_combo",
            # B1: the three per-method Refine image combos were RETIRED and
            # replaced by the ONE shared channel picker.
            "refine_method_combo", "refine_channel_combo",
            "refine_wall_kind_combo", "refine_solver_combo",
            "smooth_standalone_sigma_z_sb", "refine_prob_cmap_combo",
        ):
            w = getattr(widget, name)
            assert w.styleSheet(), "{} lost its theme".format(name)

    def test_the_sweep_is_idempotent(self, widget):
        """It only styles widgets with no stylesheet, so a second run must
        not repaint the Preview/Accept/Reject button colours."""
        before = widget.split_preview_btn.styleSheet()
        widget._apply_themed_styles()
        assert widget.split_preview_btn.styleSheet() == before

    def test_the_accept_and_reject_buttons_keep_their_own_colours(self, widget):
        """Those are set explicitly at construction and must survive both
        the sweep and the re-layout into the footer."""
        assert widget.split_accept_btn.styleSheet()
        assert widget.split_reject_btn.styleSheet()
        assert widget.add_accept_btn.styleSheet()
        assert widget.smooth_accept_btn.styleSheet()


# --------------------------------------------------------------------------- #
# The mockup's primitives, on the widgets that have to carry them
# --------------------------------------------------------------------------- #

def _squash(text) -> str:
    """A stylesheet with its whitespace removed, for colour comparison."""
    return "".join(str(text or "").split())


#: Straight out of the mockup's <style> block.
MOCK_CARD_FILL = "rgba(255,255,255,0.02)"
MOCK_CARD_BORDER = "rgba(255,255,255,0.06)"
MOCK_FIELD = "#414851"
MOCK_GOLD = "#d4a843"
MOCK_HELP = "#888888"
MOCK_WARN_FILL = "rgba(212,168,67,0.10)"
MOCK_WARN_TEXT = "#e6c26a"

#: Every card on the Reshape page: the three S3 sub-sections, the RESULT,
#: and the three Advanced groups (which are found by walking, below).
CARD_ATTRS = (
    "_split_markers_container",
    "_split_walls_group_container",
    "_split_solve_container",
    "_split_result_container",
)

#: Every row of S3 that has a label column.  They must all be the SAME
#: width -- that is what makes it a column rather than a pile of rows.
#: "Watershed on:" and "Image layer:" left this list in Brick 2b with the
#: legacy pair they labelled.
S3_ROW_LABELS = (
    "Markers from", "Channel", "Find by", "Walls from", "channel",
    "Solve with",
)


def _labels(widget):
    from qtpy.QtWidgets import QLabel

    return widget.findChildren(QLabel)


class TestMockupCards:
    @pytest.mark.parametrize("attr", CARD_ATTRS)
    def test_the_section_is_a_card(self, widget, attr):
        from napari_epicycle.ui_shell.theme import PANEL_CARD_OBJECT

        box = getattr(widget, attr)
        assert box.objectName() == PANEL_CARD_OBJECT, (
            "{} is not a `.sec` card".format(attr)
        )
        ss = _squash(box.styleSheet())
        assert MOCK_CARD_FILL in ss, "{}: card fill".format(attr)
        assert MOCK_CARD_BORDER in ss, "{}: card border".format(attr)

    @pytest.mark.parametrize("attr", CARD_ATTRS)
    def test_a_card_paints_its_background(self, widget, attr):
        """A plain QWidget ignores a stylesheet background unless it is
        told to style it -- the card would be invisible without this."""
        from qtpy.QtCore import Qt

        assert getattr(widget, attr).testAttribute(Qt.WA_StyledBackground)

    def test_the_advanced_form_is_three_cards(self, widget):
        from napari_epicycle.ui_shell.theme import PANEL_CARD_OBJECT

        cards = [
            w for w in widget._split_advanced_container.findChildren(
                type(widget._split_advanced_container))
            if w.objectName() == PANEL_CARD_OBJECT
        ]
        assert len(cards) == 3, (
            "Advanced is {} card(s); the mockup groups it into three "
            "(Markers / Walls / Cut and snap)".format(len(cards))
        )


class TestMockupHeaders:
    def _headers(self, widget):
        return [
            lb for lb in _labels(widget)
            if _squash(lb.styleSheet()).startswith("color:#999999;font-size:10px")
        ]

    def test_the_result_card_has_an_uppercase_header(self, widget):
        texts = [lb.text() for lb in self._headers(widget)]
        assert "RESULT" in texts, texts

    def test_the_advanced_groups_are_headed(self, widget):
        texts = [lb.text() for lb in self._headers(widget)]
        for want in ("MARKERS", "WALLS", "CUT AND SNAP"):
            assert want in texts, "{} missing from {}".format(want, texts)

    def test_a_header_is_letter_spaced(self, widget):
        """Qt stylesheets have no letter-spacing, so it has to be on the
        QFont -- and a QSS font-size must not have wiped it."""
        headers = [lb for lb in self._headers(widget) if lb.text() == "RESULT"]
        assert headers, "no RESULT header"
        assert headers[0].font().letterSpacing() != 0


class TestMockupLabelColumn:
    def test_every_s3_row_shares_one_label_column(self, widget):
        seen = {}
        for lb in _labels(widget):
            if lb.text() in S3_ROW_LABELS:
                seen.setdefault(lb.text(), lb.width() or lb.minimumWidth())
        missing = set(S3_ROW_LABELS) - set(seen)
        assert not missing, "no label found for {}".format(sorted(missing))
        widths = set(seen.values())
        assert len(widths) == 1, (
            "the S3 label column is ragged: {}".format(seen)
        )

    def test_the_column_is_the_one_the_theme_defines(self, widget):
        from napari_epicycle.ui_shell.theme import PANEL_LABEL_COLUMN

        for lb in _labels(widget):
            if lb.text() == "Markers from":
                assert lb.minimumWidth() == PANEL_LABEL_COLUMN
                assert lb.maximumWidth() == PANEL_LABEL_COLUMN
                return
        raise AssertionError("the Markers from row lost its label")

    def test_the_advanced_knobs_share_their_own_wider_column(self, widget):
        from napari_epicycle.ui_shell.theme import PANEL_ADV_LABEL_COLUMN

        seen = {}
        for lb in widget._split_advanced_container.findChildren(type(
                widget.split_status_label)):
            # Brick 2b relabelled three of these and promoted the
            # fourth ("force piece count") into How-to-cut, where it is
            # a tick rather than a label in this column.
            if lb.text() in ("Prominence", "Wall clip",
                             "Soft cut: wall resistance", "Balance"):
                seen[lb.text()] = lb.minimumWidth()
        assert len(seen) == 4, seen
        assert set(seen.values()) == {PANEL_ADV_LABEL_COLUMN}, seen


class TestMockupChipsAndChevrons:
    @pytest.mark.parametrize("attr", [
        "split_seed_prep_btn", "split_wall_prep_btn",
        "split_marker_all_btn", "split_marker_none_btn",
    ])
    def test_the_quick_toggles_are_chips(self, widget, attr):
        from napari_epicycle.ui_shell.theme import PANEL_CHIP_STYLE

        assert getattr(widget, attr).styleSheet() == PANEL_CHIP_STYLE

    def test_a_chip_tints_gold_while_it_is_open(self, widget):
        ss = _squash(widget.split_seed_prep_btn.styleSheet())
        assert "QPushButton:checked{background-color:rgba(212,168,67,0.20)" in ss

    @pytest.mark.parametrize("attr", [
        "split_guided_summary_btn", "split_markers_btn", "split_walls_btn",
        "split_solve_btn", "split_advanced_btn",
    ])
    def test_the_disclosures_are_left_aligned_rows(self, widget, attr):
        ss = _squash(getattr(widget, attr).styleSheet())
        assert "text-align:left" in ss, attr


class TestMockupListRows:
    @pytest.mark.parametrize("attr", ["split_marker_list", "split_piece_list"])
    def test_the_ticklist_rows_are_pieces(self, widget, attr):
        from napari_epicycle.ui_shell.theme import PanelRowDelegate

        lst = getattr(widget, attr)
        assert lst.iconSize().width() == 12, "the `.sw` swatch is 12 px"
        assert isinstance(lst.itemDelegate(), PanelRowDelegate), (
            "{} does not draw its trailing value in mono".format(attr)
        )

    @pytest.mark.parametrize("attr", ["split_marker_list", "split_piece_list"])
    def test_a_ticked_row_is_gold(self, widget, attr):
        ss = _squash(getattr(widget, attr).styleSheet())
        assert "QListWidget::indicator:checked" in ss
        assert MOCK_GOLD in ss

    def test_the_delegate_leaves_the_row_text_alone(self, widget):
        """It only decides which part of the text is drawn in which font;
        every test that reads a row still reads the same string."""
        widget._populate_marker_list(
            [1, 2], {1: {"voxels": 1204}, 2: {"voxels": 980}})
        assert widget.split_marker_list.item(0).text().startswith("Marker 1")
        assert "1,204" in widget.split_marker_list.item(0).text()

    def test_a_list_hugs_its_rows(self, widget):
        """The mockup's `.list` is as tall as what is in it; a fixed
        reserve leaves a hole under a short result."""
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.update_split_guided_ui(
            pieces=[11, 12], piece_sizes={11: 4000, 12: 3800})
        assert (widget.split_piece_list.maximumHeight()
                < M._PIECE_LIST_MAX_H)


class TestMockupHelpWarnAndStatus:
    def test_the_helper_lines_are_the_mockup_grey(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        assert MOCK_HELP in _squash(M._HELPER_LINE_STYLE)
        assert "font-size:10px" in _squash(M._HELPER_LINE_STYLE)

    @pytest.mark.parametrize("attr", [
        "split_marker_caption", "split_piece_caption",
        "split_marker_count_label", "split_engine_msg_label",
    ])
    def test_every_grey_line_uses_it(self, widget, attr):
        from napari_epicycle.annotator import advanced_features_widget as M

        assert getattr(widget, attr).styleSheet() == M._HELPER_LINE_STYLE

    @pytest.mark.parametrize("attr", [
        "split_arbiter_msg_label", "split_guide_note_label",
    ])
    def test_the_warnings_are_the_amber_block(self, widget, attr):
        ss = _squash(getattr(widget, attr).styleSheet())
        assert MOCK_WARN_FILL in ss, attr
        assert MOCK_WARN_TEXT in ss, attr

    @pytest.mark.parametrize("attr", [
        "_split_s1_container", "_add_s1_container", "_refine_s1_container",
    ])
    def test_the_status_line_is_a_pill(self, widget, attr):
        from napari_epicycle.ui_shell.theme import PANEL_STATUS_PILL_OBJECT

        row = getattr(widget, attr)
        assert row.objectName() == PANEL_STATUS_PILL_OBJECT
        assert "border-radius" in _squash(row.styleSheet())

    @pytest.mark.parametrize("name", ["Reshape", "Refine", "Add"])
    def test_the_footer_is_a_footer(self, widget, name):
        from napari_epicycle.ui_shell.theme import PANEL_FOOTER_OBJECT

        footer = widget._panel_slots[name]["footer"]
        assert footer.objectName() == PANEL_FOOTER_OBJECT
        assert "border-top" in _squash(footer.styleSheet())


class TestMockupAccentAndFields:
    @pytest.mark.parametrize("attr", ["split_preview_btn", "split_accept_btn"])
    def test_the_committing_actions_are_gold(self, widget, attr):
        from napari_epicycle.ui_shell.theme import PANEL_PRIMARY_BTN_STYLE

        assert getattr(widget, attr).styleSheet() == PANEL_PRIMARY_BTN_STYLE
        assert MOCK_GOLD in _squash(PANEL_PRIMARY_BTN_STYLE)

    def test_every_field_is_the_mockup_grey(self, widget):
        # superqt's SliderLabel arrives with a stylesheet of its own (it
        # is the number glued to a slider handle, not a field), so the
        # sweep leaves it alone and so does this.
        wrong = [
            type(w).__name__ for w in _inputs(widget)
            if type(w).__name__ != "SliderLabel"
            and MOCK_FIELD not in _squash(w.styleSheet())
        ]
        assert not wrong, "{} input(s) are not on {}".format(
            len(wrong), MOCK_FIELD)

    def test_the_ticks_are_gold_when_they_are_on(self, widget):
        """Set on the panel ROOT so it reaches every tick, including the
        ones built inside a stacked page."""
        ss = _squash(widget.styleSheet())
        assert "QCheckBox::indicator:checked" in ss
        assert MOCK_GOLD in ss


# --------------------------------------------------------------------------- #
# The tick INSIDE the box, and the surface this design does NOT own
# --------------------------------------------------------------------------- #

def _first_row_tick(lst):
    """``(gold, dark)`` pixel counts for the first row's check indicator.

    The moment a stylesheet touches ``::indicator`` Qt stops drawing its
    own glyph, so "the box is gold" is NOT evidence that anything is
    ticked -- the failure mode is a solid gold square.  This renders the
    live list, finds the gold box on its first row, and counts the dark
    pixels INSIDE that box: the tick, or nothing.
    """
    img = lst.grab().toImage()
    row = lst.visualItemRect(lst.item(0))
    gold = []
    for y in range(max(0, row.top()), min(img.height(), row.bottom() + 1)):
        for x in range(0, min(40, img.width())):
            c = img.pixelColor(x, y)
            if c.red() > 150 and 110 < c.green() < 210 and c.blue() < 120:
                gold.append((x, y))
    if not gold:
        return 0, 0
    xs = [p[0] for p in gold]
    ys = [p[1] for p in gold]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dark = 0
    for y in range(y0 + 1, y1):
        for x in range(x0 + 1, x1):
            c = img.pixelColor(x, y)
            if c.red() < 100 and c.green() < 100 and c.blue() < 100:
                dark += 1
    return len(gold), dark


class TestTickGlyph:
    def _markers_shown(self, widget, qapp):
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.resize(340, 700)
        widget.show()
        qapp.processEvents()
        widget.focus_tab("Reshape")
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        widget.update_split_ui_state(
            split_active=True, marker_ids=[1, 2],
            marker_stats={1: {"voxels": 1204}, 2: {"voxels": 980}})
        qapp.processEvents()
        return widget.split_marker_list

    def test_the_tick_image_is_a_file_that_exists(self):
        """A `data:` URI loads nothing at all in a Qt stylesheet, so the
        glyph has to come from a file -- which a wheel then has to ship."""
        import os

        from napari_epicycle.ui_shell.theme import (
            PANEL_CHECKBOX_STYLE, _PANEL_TICK_SVG,
        )

        assert os.path.isfile(_PANEL_TICK_SVG)
        assert _PANEL_TICK_SVG in PANEL_CHECKBOX_STYLE
        assert "image:url(" in _squash(PANEL_CHECKBOX_STYLE)

    def test_a_ticked_box_actually_has_a_tick_in_it(self, widget, qapp):
        lst = self._markers_shown(widget, qapp)
        try:
            gold, dark = _first_row_tick(lst)
            assert gold > 20, (
                "the ticked row's box is not gold ({} px)".format(gold))
            assert dark > 0, (
                "the gold box is solid -- Qt drew no tick in it"
            )
        finally:
            widget.hide()

    def test_the_measurement_can_tell_a_solid_box_apart(self, widget, qapp):
        """The control: strip the image rule and the same measurement
        must find NOTHING, or the test above proves nothing."""
        from napari_epicycle.ui_shell import theme as T

        lst = self._markers_shown(widget, qapp)
        try:
            url = 'image: url("{}");'.format(T._PANEL_TICK_SVG)
            lst.setStyleSheet(T.PANEL_LIST_STYLE.replace(url, ""))
            widget.setStyleSheet(T.PANEL_CHECKBOX_STYLE.replace(url, ""))
            qapp.processEvents()
            _gold, dark = _first_row_tick(lst)
            assert dark == 0, (
                "the measurement finds a tick where there is none"
            )
        finally:
            widget.hide()


class TestTheHitl2dSurfaceIsUntouched:
    """The Brick 2 mockup is the STEP 3 dock's design.  The HITL 2D
    annotator is a different surface from the 3D one, so the same widget class
    in ``mode="hitl_2d"`` must come out exactly as it did before."""

    @pytest.fixture
    def hitl(self, qapp):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        w = AdvancedFeaturesWidget(mode="hitl_2d")
        yield w
        try:
            w.setParent(None)
        except Exception:
            pass

    def test_no_root_stylesheet(self, hitl):
        assert hitl.styleSheet() == ""

    def test_the_inputs_keep_the_shared_theme(self, hitl):
        from napari_epicycle.ui_shell.theme import INPUT_STYLE

        inputs = _inputs(hitl)
        assert inputs
        for w in inputs:
            if type(w).__name__ == "SliderLabel":
                continue
            assert w.styleSheet() == INPUT_STYLE, type(w).__name__

    def test_the_inputs_keep_their_height(self, hitl):
        from qtpy.QtWidgets import QComboBox

        combos = hitl.findChildren(QComboBox)
        assert combos
        assert combos[0].minimumSizeHint().height() == 26

    def test_the_numbers_are_not_re_aligned(self, hitl):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDoubleSpinBox, QSpinBox

        for cls in (QSpinBox, QDoubleSpinBox):
            for w in hitl.findChildren(cls):
                assert not (w.alignment() & Qt.AlignRight), type(w).__name__


# --------------------------------------------------------------------------- #
# A shrinkable spin box still has to show its number (owner, 2026-09-18)
# --------------------------------------------------------------------------- #
class TestSpinBoxesStayReadable:
    """`_make_widths_shrinkable` gives every spin box ``Ignored``, which
    means a minimum of ZERO -- and on the Detect row the layout took
    that literally: "Min dist" rendered as a bare minus button with no
    value in it at all.  Every box must be at least as wide as the
    number it can hold, and the page must still fit its viewport."""

    def _shown_with_every_knob(self, widget, qapp):
        widget.resize(340, 350)
        widget.show()
        qapp.processEvents()
        widget.focus_tab("Reshape")
        widget.update_split_ui_state(
            split_active=True, detect_visible=True, detect_seed_count=4)
        widget.split_advanced_btn.setChecked(True)
        widget.split_guided_summary_btn.setChecked(True)
        qapp.processEvents()
        qapp.processEvents()
        return widget._panel_slots["Reshape"]

    def _spins(self, root):
        from qtpy.QtWidgets import QDoubleSpinBox, QSpinBox

        out = []
        for cls in (QSpinBox, QDoubleSpinBox):
            out.extend(w for w in root.findChildren(cls) if not w.isHidden())
        return out

    def test_no_spin_box_is_starved_of_its_own_value(self, widget, qapp):
        slots = self._shown_with_every_knob(widget, qapp)
        try:
            boxes = self._spins(slots["content"])
            assert boxes
            starved = []
            for box in boxes:
                need = box.fontMetrics().horizontalAdvance(
                    str(box.maximum())) + 30
                if box.width() < need:
                    starved.append((type(box).__name__, box.width(), need))
            assert not starved, (
                "{} spin box(es) too narrow to show their value: "
                "{}".format(len(starved), starved)
            )
        finally:
            widget.hide()

    def test_the_detect_row_is_one_knob_per_row(self, widget, qapp):
        """Both on one row is what starved them; per-row also puts them
        on the same label column as the rest of the page."""
        slots = self._shown_with_every_knob(widget, qapp)
        try:
            a = widget.split_detect_min_dist_sb
            b = widget.split_detect_bins_sb
            assert a.x() == b.x(), "the two knobs share a left edge"
            assert a.y() != b.y(), "they are on separate rows"
            assert a.width() > 100, a.width()
        finally:
            widget.hide()

    def test_the_page_still_fits_its_viewport(self, widget, qapp):
        slots = self._shown_with_every_knob(widget, qapp)
        try:
            assert (slots["content"].minimumSizeHint().width()
                    <= slots["scroll"].viewport().width())
        finally:
            widget.hide()

    def test_a_minimum_is_the_boxs_own_font(self, widget):
        """Not a magic 64: it scales with the machine's font."""
        box = widget.split_rw_beta_sb
        need = box.fontMetrics().horizontalAdvance("1000.00")
        assert box.minimumWidth() >= need


# --------------------------------------------------------------------------- #
# The ticklists are resizable (owner: "feels so cramped if there are many")
# --------------------------------------------------------------------------- #
GRIPPED = [
    ("split_marker_list", "split_marker_list_grip", "_MARKER_LIST_H_PROP",
     "_MARKER_LIST_MAX_H"),
    ("split_piece_list", "split_piece_list_grip", "_PIECE_LIST_H_PROP",
     "_PIECE_LIST_MAX_H"),
]


class TestListHeightGrip:
    def _filled(self, widget, qapp, pieces=True):
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.resize(340, 700)
        widget.show()
        qapp.processEvents()
        widget.focus_tab("Reshape")
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        # the Find row (and its list) belong to the image-channel source
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        widget.update_split_ui_state(
            split_active=True, marker_ids=[1, 2, 3, 4, 5],
            marker_stats={i: 1000 * i for i in range(1, 6)})
        if pieces:
            widget.update_split_guided_ui(
                pieces=[11, 12, 13],
                piece_sizes={11: 4000, 12: 3800, 13: 2000})
        qapp.processEvents()

    @pytest.mark.parametrize("list_attr,grip_attr,_prop,_cap", GRIPPED)
    def test_the_list_has_a_grip_under_it(self, widget, list_attr, grip_attr,
                                          _prop, _cap):
        from napari_epicycle.ui_shell.theme import (
            PANEL_GRIP_H, PANEL_GRIP_STYLE,
        )
        from qtpy.QtCore import Qt

        grip = getattr(widget, grip_attr)
        assert grip.height() == PANEL_GRIP_H
        assert grip.styleSheet() == PANEL_GRIP_STYLE
        assert grip.cursor().shape() == Qt.SizeVerCursor
        assert grip.toolTip().strip()
        # ... and it is a child of the panel, like everything else here
        node, seen = grip, False
        while node is not None:
            if node is widget:
                seen = True
                break
            node = node.parentWidget()
        assert seen

    @pytest.mark.parametrize("list_attr,grip_attr,_prop,_cap", GRIPPED)
    def test_the_grip_hides_with_its_list(self, widget, qapp, list_attr,
                                          grip_attr, _prop, _cap):
        grip = getattr(widget, grip_attr)
        lst = getattr(widget, list_attr)
        # One at a time: a review collapses the markers with Sources, so
        # the two ticklists are never on screen together (inv. 9).
        self._filled(widget, qapp, pieces=list_attr == "split_piece_list")
        try:
            assert lst.isVisible()
            assert grip.isVisible()
            lst.setVisible(False)
            qapp.processEvents()
            assert not grip.isVisible(), (
                "a handle under a hidden list is a handle to nothing"
            )
        finally:
            widget.hide()

    @pytest.mark.parametrize("list_attr,grip_attr,prop,cap", GRIPPED)
    def test_dragging_resizes_within_the_bounds(self, widget, qapp,
                                                list_attr, grip_attr, prop,
                                                cap):
        from napari_epicycle.annotator import advanced_features_widget as M

        self._filled(widget, qapp)
        try:
            lst = getattr(widget, list_attr)
            grip = getattr(widget, grip_attr)
            low, high = widget._list_height_bounds(lst)
            assert low < high

            before = lst.maximumHeight()
            grip.drag_by(60)
            qapp.processEvents()
            assert lst.maximumHeight() > before
            assert low <= lst.maximumHeight() <= high
            assert widget.property(getattr(M, prop)) == lst.maximumHeight()

            grip.drag_by(10000)
            assert lst.maximumHeight() == high, "the top of the range"
            grip.drag_by(-10000)
            assert lst.maximumHeight() == low, "the bottom of the range"
        finally:
            widget.hide()

    @pytest.mark.parametrize("list_attr,grip_attr,prop,cap", GRIPPED)
    def test_the_height_survives_a_refill(self, widget, qapp, list_attr,
                                          grip_attr, prop, cap):
        """A Find, a Preview or a Discard must not undo the drag."""
        self._filled(widget, qapp)
        try:
            lst = getattr(widget, list_attr)
            grip = getattr(widget, grip_attr)
            grip.drag_by(50)
            chosen = lst.maximumHeight()
            widget._populate_marker_list(
                [1, 2], {1: {"voxels": 10}, 2: {"voxels": 20}})
            widget.update_split_guided_ui(
                pieces=[21, 22], piece_sizes={21: 10, 22: 20})
            qapp.processEvents()
            assert lst.maximumHeight() == chosen
        finally:
            widget.hide()

    @pytest.mark.parametrize("list_attr,grip_attr,prop,cap", GRIPPED)
    def test_a_double_click_hugs_the_rows_again(self, widget, qapp,
                                                list_attr, grip_attr, prop,
                                                cap):
        from napari_epicycle.annotator import advanced_features_widget as M

        self._filled(widget, qapp)
        try:
            lst = getattr(widget, list_attr)
            grip = getattr(widget, grip_attr)
            default = lst.maximumHeight()
            grip.drag_by(70)
            assert lst.maximumHeight() != default
            grip.reset()
            qapp.processEvents()
            assert widget.property(getattr(M, prop)) in (None, 0)
            assert lst.minimumHeight() == 0
            assert lst.maximumHeight() == default
        finally:
            widget.hide()

    def test_the_handle_is_paid_for_out_of_the_lists_cap(self):
        """It must not cost the PAGE anything: markers-found has 7 px of
        headroom against its two-screen ceiling."""
        from napari_epicycle.annotator import advanced_features_widget as M
        from napari_epicycle.ui_shell.theme import PANEL_GRIP_H

        assert M._MARKER_LIST_MAX_H + PANEL_GRIP_H == 90
        assert M._PIECE_LIST_MAX_H + PANEL_GRIP_H == 130


class TestTheRefusalStyle:
    """M8: S5 says "the engine message (refusals red)", and
    the colour rule is amber = a warning you may ignore, red = the
    tool did not do what you asked.  Both were amber-on-grey: a refusal
    ("Pick the image channel the walls are in") was rendered in the same
    grey as the informational "Wall contrast 0.12-0.81" line under it.
    """

    def test_the_theme_defines_it(self):
        from napari_epicycle.ui_shell import theme

        assert hasattr(theme, "PANEL_ERROR_STYLE")

    def test_it_is_red_and_not_the_amber_one(self):
        from napari_epicycle.ui_shell.theme import (
            PANEL_ERROR_STYLE, PANEL_WARN_STYLE,
        )

        assert PANEL_ERROR_STYLE != PANEL_WARN_STYLE
        low = PANEL_ERROR_STYLE.lower()
        assert "224, 87, 79" in low or "#e0" in low, PANEL_ERROR_STYLE

    def test_it_is_built_like_the_amber_one(self):
        """Same block shape -- background, border, radius, padding, size
        -- so the two read as one family and differ only in colour."""
        from napari_epicycle.ui_shell.theme import (
            PANEL_ERROR_STYLE, PANEL_WARN_STYLE,
        )

        for prop in ("background", "border", "border-radius", "padding",
                     "font-size"):
            assert prop in PANEL_ERROR_STYLE, prop
            assert prop in PANEL_WARN_STYLE, prop

    def test_the_refusal_label_wears_it(self, widget):
        from napari_epicycle.ui_shell.theme import PANEL_ERROR_STYLE

        assert widget.split_engine_error_label.styleSheet() == (
            PANEL_ERROR_STYLE)

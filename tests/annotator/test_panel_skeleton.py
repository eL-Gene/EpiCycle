"""The S1-S6 skeleton: scroll area, fixed footer, and nothing appended.

Spec invariants pinned here:

* **I-4** no widget is added to a page layout after construction -- every
  stage change is ``setVisible`` on a container built in the builder.  The
  two documented exceptions are the pyqtgraph placeholder swaps.
* **I-6** the RESULT is never below the footer; the footer is a descendant
  of its own tab page and lives OUTSIDE the scroll area; ADVANCED is the
  last thing INSIDE the scroll content.
* **I-8** two measured ceilings, not one: the panel's minimum height and
  the footer's own height (a single global ceiling passes happily while
  the footer has eaten the viewport).
* **I-13** a wheel over an unfocused spin box scrolls the page instead of
  silently changing the parameter.
* **I-14** every stage swap puts S1 back in view and focus on Preview, and
  never leaves focus inside a hidden container.

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


VISIBLE_PAGES = ("Reshape", "Refine", "Add")


def _slots(widget, name):
    return widget._panel_slots[name]


def _page_of(widget, name):
    tabs = widget._tool_tabs
    for i in range(tabs.count()):
        if tabs.tabText(i) == name:
            return tabs.widget(i)
    raise AssertionError("no tab named {}".format(name))


def _layout_fingerprint(root):
    """Recursive (widget-class, item-count) fingerprint of a layout tree."""
    out = []

    def walk(layout, depth):
        if layout is None:
            return
        out.append((depth, layout.count()))
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget()
            if w is not None:
                out.append((depth, type(w).__name__))
                walk(w.layout(), depth + 1)
            else:
                walk(item.layout(), depth + 1)

    walk(root.layout(), 0)
    return out


# --------------------------------------------------------------------------- #
# I-6 -- scroll inside the page, footer beside it, result above it
# --------------------------------------------------------------------------- #
class TestPageSkeleton:
    #: Pages that pin one action strip between the scroll and the
    #: footer.  Reshape does, since 2026-09-19: Preview is needed in
    #: every state and S3 can be taller than the whole viewport, so a
    #: scrolling Preview is a Preview the user cannot reach (measured
    #: live at y=395 in a 207 px window).  Any page added here owes the
    #: same justification -- the strip is paid for out of the scroll.
    #: Refine joined it in Brick B1, for the same
    #: reason and with one addition: its strip carries a Stop button and a
    #: stage line as well as the bar, so all THREE slots are reserved at
    #: construction and `TestTheRefineStripDoesNotMove` asserts its height
    #: is the same in idle, working and staged.
    #: Add joined it in the Add rebuild: the Cells
    #: preset's Markers + Walls cards make S3 taller than the viewport, and
    #: its strip carries Stop and a stage line like Refine's.
    PAGES_WITH_A_PINNED_ACTION = ("Reshape", "Refine", "Add")

    @pytest.mark.parametrize("name", VISIBLE_PAGES)
    def test_a_page_is_a_scroll_area_plus_a_footer(self, widget, name):
        from qtpy.QtWidgets import QScrollArea

        page = _page_of(widget, name)
        s = _slots(widget, name)
        assert s["page"] is page
        assert isinstance(s["scroll"], QScrollArea)
        lay = page.layout()
        pinned = 1 if name in self.PAGES_WITH_A_PINNED_ACTION else 0
        assert lay.count() == 2 + pinned, (
            "a page is the scroll area, an optional pinned action strip, "
            "and the footer -- in that order"
        )
        # The order is the contract, not the count: the scroll takes the
        # slack, anything pinned sits under it, the footer is last.
        assert lay.indexOf(s["scroll"]) == 0
        assert lay.indexOf(s["footer"]) == lay.count() - 1

    @pytest.mark.parametrize("name", VISIBLE_PAGES)
    def test_the_scroll_area_is_resizable_and_frameless(self, widget, name):
        from qtpy.QtWidgets import QFrame

        s = _slots(widget, name)
        assert s["scroll"].widgetResizable() is True
        assert s["scroll"].frameShape() == QFrame.NoFrame
        assert s["scroll"].widget() is s["content"]

    @pytest.mark.parametrize("name", VISIBLE_PAGES)
    def test_the_footer_belongs_to_its_page_and_not_to_the_scroll(
            self, widget, name):
        s = _slots(widget, name)
        # a descendant of the page ...
        node, seen_page = s["footer"], False
        while node is not None:
            if node is s["page"]:
                seen_page = True
                break
            node = node.parentWidget()
        assert seen_page, "the footer must be a descendant of its own page"
        # ... and NOT of the scroll content
        node, in_scroll = s["footer"], False
        while node is not None:
            if node is s["content"]:
                in_scroll = True
                break
            node = node.parentWidget()
        assert not in_scroll, "the footer must not scroll away"

    @pytest.mark.parametrize("name", VISIBLE_PAGES)
    def test_the_footer_is_two_buttons_and_a_summary_line(self, widget, name):
        from qtpy.QtWidgets import QLabel, QPushButton

        s = _slots(widget, name)
        buttons = s["footer"].findChildren(QPushButton)
        assert len(buttons) == 2, (
            "{} footer holds {} buttons; the contract is exactly two".format(
                name, len(buttons))
        )
        assert s["footer"].findChildren(QLabel), (
            "the footer carries the one-line summary"
        )

    @pytest.mark.parametrize("name,attr", [
        ("Reshape", "_split_footer_summary"),
        ("Add", "_add_footer_summary"),
        ("Refine", "_refine_footer_summary"),
    ])
    def test_the_summary_line_is_never_blank(self, widget, name, attr):
        """An empty line is a slot the layout reserves and says nothing in;
        all three were blank on two of the three tabs."""
        assert getattr(widget, attr).text().strip(), (
            "{}'s footer line is empty at birth".format(name)
        )

    def test_every_footer_line_tracks_its_own_states(self, widget):
        seen = {"Reshape": set(), "Add": set(), "Refine": set()}

        def _snap():
            seen["Reshape"].add(widget._split_footer_summary.text())
            seen["Add"].add(widget._add_footer_summary.text())
            seen["Refine"].add(widget._refine_footer_summary.text())

        _snap()
        widget.update_split_ui_state(reject_text="Discard")
        widget.update_add_ui_state(
            candidate_ids=[3], candidate_sizes={3: 900})
        widget.update_smooth_ui_state(reject_enabled=True)
        _snap()
        widget.update_add_ui_state(review_active=True, piece_ids=[3])
        _snap()
        for tab, texts in seen.items():
            assert all(t.strip() for t in texts), tab
            assert len(texts) >= 2, (
                "{}'s footer line says the same thing in every state".format(
                    tab)
            )

    def test_the_result_section_sits_above_the_footer(self, widget):
        s = _slots(widget, "Reshape")
        content_layout = s["content"].layout()
        i_result = content_layout.indexOf(widget._split_result_container)
        assert i_result >= 0, "S5 is a direct child of the scroll content"
        # S5 is inside the scroll and the footer is after it: above by
        # construction.  S4 left the scroll content (it is pinned between
        # the scroll and the footer), so the ordering that matters now is
        # the one on the PAGE.
        page_layout = s["page"].layout()
        assert page_layout.indexOf(s["scroll"]) < page_layout.indexOf(
            widget._split_s4_container)

    def test_advanced_is_the_last_thing_inside_the_scroll_content(self, widget):
        s = _slots(widget, "Reshape")
        node, inside = widget._split_advanced_container, False
        while node is not None:
            if node is s["content"]:
                inside = True
                break
            node = node.parentWidget()
        assert inside, "ADVANCED must scroll, not pin the viewport open"
        content_layout = s["content"].layout()
        i_adv = content_layout.indexOf(widget._split_advanced_container)
        i_result = content_layout.indexOf(widget._split_result_container)
        assert i_adv > i_result

    def test_advanced_is_one_disclosure_deep_not_two(self, widget):
        """Finding 3: the Advanced form used to live INSIDE the guided
        disclosure, so reaching a solver knob took two clicks and inserted
        ~18 rows mid-page."""
        node = widget._split_advanced_container.parentWidget()
        while node is not None:
            assert node is not widget._split_guided_container, (
                "ADVANCED is still nested inside the SOURCES disclosure"
            )
            node = node.parentWidget()

    def test_the_stop_button_sits_beside_preview(self, widget):
        """Finding 8: 'Stop waiting' used to be five rows ABOVE the Preview
        button that starts the work it stops."""
        row = widget.split_cancel_preview_btn.parentWidget()
        assert row is widget.split_preview_btn.parentWidget()
        node, inside = widget.split_preview_btn, False
        while node is not None:
            if node is widget._split_s4_container:
                inside = True
                break
            node = node.parentWidget()
        assert inside

    def test_napari_cannot_inject_its_stretch(self, widget):
        """napari's _maybe_add_vertical_stretch skips a root child whose
        vertical policy is Expanding; QTabWidget defaults to Preferred."""
        from qtpy.QtWidgets import QSizePolicy

        assert widget._tool_tabs.sizePolicy().verticalPolicy() in (
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding,
            QSizePolicy.Ignored,
        )
        for name in VISIBLE_PAGES:
            assert _slots(widget, name)["scroll"].sizePolicy().verticalPolicy() in (
                QSizePolicy.Expanding, QSizePolicy.MinimumExpanding,
                QSizePolicy.Ignored,
            )


# --------------------------------------------------------------------------- #
# I-4 -- nothing is appended after construction
# --------------------------------------------------------------------------- #
class TestNothingIsAppendedAfterConstruction:
    def _drive(self, widget):
        """Walk the Reshape + Add stages that a session goes through."""
        widget.update_split_ui_state(
            split_active=True, seed_count=1, seed_ids=[9],
            preview_enabled=True, reject_enabled=True, reject_text="Discard",
            status_text="Cell 149 is ready.",
        )
        widget.update_split_ui_state(detect_visible=True, detect_seed_count=4)
        widget.update_split_guided_ui(busy=True, busy_progress=(2, 8))
        widget.update_split_guided_ui(busy=False)
        widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 4000, 12: 3800, 13: 90},
            caption="unticked pieces stay part of cell 149",
            arbiter_message="kept whole", split_anyway_visible=True,
            engine_messages=["used watershed instead"],
        )
        widget.update_split_ui_state(
            fragment_refine_active=True, fragment_ids=[11, 12],
            active_fragment_id=11,
        )
        widget.clear_split_guided_ui()
        widget.update_split_ui_state(
            split_active=False, fragment_refine_active=False,
            detect_visible=False, reject_text="Cancel",
        )
        widget.update_add_ui_state(
            box_active=True, box_shape=(40, 96, 96), box_center=(20, 48, 48),
            box_size=(10, 20, 20), preview_enabled=True,
            candidate_ids=[3, 4], candidate_sizes={3: 900, 4: 40},
        )
        widget.update_add_ui_state(review_active=True, piece_ids=[3])
        widget.update_add_ui_state(review_active=False)
        for i in range(widget.refine_method_combo.count()):
            widget.refine_method_combo.setCurrentIndex(i)

    @pytest.mark.parametrize("name", VISIBLE_PAGES)
    def test_the_layout_tree_is_identical_before_and_after(self, widget, name):
        page = _page_of(widget, name)
        before = _layout_fingerprint(page)
        self._drive(widget)
        after = _layout_fingerprint(page)
        assert before == after, (
            "{} grew or shrank a layout at runtime -- a stage change must "
            "only flip container visibility".format(name)
        )

    def test_the_two_plot_swaps_are_the_only_exemption(self, widget):
        """`_ensure_detect_plot` / `_ensure_add_plot` replace a placeholder
        with the live pyqtgraph widget 1-for-1, in ONE container."""
        pytest.importorskip("pyqtgraph")
        holder = widget._detect_plot_placeholder.parentWidget()
        before = holder.layout().count()
        widget._ensure_detect_plot()
        assert holder.layout().count() == before, "must be a 1-for-1 swap"
        assert widget._detect_plot_placeholder is None
        assert widget._detect_plot_widget is not None
        assert widget._detect_plot_widget.parentWidget() is holder, (
            "the placeholder and the live plot must share a parent so the "
            "swap stays local to one container"
        )


# --------------------------------------------------------------------------- #
# I-8 -- two ceilings
# --------------------------------------------------------------------------- #
class TestHeightCeilings:
    def _states(self, widget):
        yield "idle", lambda: None
        yield "seeding", lambda: widget.update_split_ui_state(
            split_active=True, seed_ids=[9, 10], seed_count=2)
        yield "solving", lambda: widget.update_split_guided_ui(busy=True)
        yield "review", lambda: widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 4000, 12: 3800, 13: 2000},
            caption="unticked pieces stay part of cell 149",
            arbiter_message="some cuts had no membrane behind them",
            split_anyway_visible=True,
            engine_messages=["u-Segment3D is not installed"],
        )
        yield "sources+advanced open", lambda: (
            widget.split_guided_summary_btn.setChecked(True),
            widget.split_advanced_btn.setChecked(True),
        )

    def test_the_panel_minimum_fits_the_dock_the_launcher_opens(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        for label, step in self._states(widget):
            step()
            h = widget.minimumSizeHint().height()
            assert h <= M._MAX_PANEL_H, (
                "state {!r}: panel minimum {}px > {}px, so the 340x350 dock "
                "the launcher opens cannot honour it".format(
                    label, h, M._MAX_PANEL_H)
            )

    def _drive_every_footer_state(self, widget):
        """Yield (tag) after each push that can change a footer line."""
        from qtpy.QtWidgets import QApplication

        yield "idle"
        widget.update_split_ui_state(
            split_active=True, reject_text="Discard", preview_enabled=True)
        yield "reshape staged"
        widget.update_split_guided_ui(busy=True)
        yield "solving"
        widget.update_split_guided_ui(
            busy=False, pieces=[11, 12, 13],
            piece_sizes={11: 4000, 12: 3800, 13: 2000},
            caption="unticked pieces stay part of cell 149")
        widget.update_split_ui_state(review_active=True, accept_enabled=True)
        yield "reshape review"
        widget.update_add_ui_state(
            candidate_ids=[3, 4], candidate_sizes={3: 900, 4: 800})
        yield "add candidates"
        widget.update_add_ui_state(review_active=True, piece_ids=[3, 4])
        yield "add review"
        widget.update_smooth_ui_state(reject_enabled=True)
        yield "refine staged"
        QApplication.processEvents()

    def test_the_footer_height_is_CONSTANT_across_states(self, widget):
        """The live driver measured 54 px -> 69 px: the summary line wraps
        to two lines for the longer states and one for the shorter ones, so
        the one thing that must not move moved on every stage change (and
        took the dock's minimum height with it)."""
        from qtpy.QtWidgets import QApplication

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        try:
            for page in VISIBLE_PAGES:
                widget.focus_tab(page)
                QApplication.processEvents()
                footer = _slots(widget, page)["footer"]
                heights = set()
                for _tag in self._drive_every_footer_state(widget):
                    QApplication.processEvents()
                    heights.add(footer.height())
                assert len(heights) == 1, (
                    "{} footer moved across states: {}".format(
                        page, sorted(heights))
                )
        finally:
            widget.hide()

    def test_the_panel_minimum_does_not_grow_with_the_stage(self, widget):
        from qtpy.QtWidgets import QApplication

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        try:
            mins = set()
            for _tag in self._drive_every_footer_state(widget):
                QApplication.processEvents()
                mins.add(widget.minimumSizeHint().height())
            assert len(mins) == 1, (
                "the dock's minimum grew as the stage changed: {}".format(
                    sorted(mins))
            )
        finally:
            widget.hide()

    def test_no_footer_summary_text_is_clipped(self, widget):
        """The slot is reserved at two lines; every line it can show has to
        FIT in two lines, or the reservation just hides the second half."""
        from qtpy.QtWidgets import QApplication, QLabel

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        labels = (
            widget._split_footer_summary,
            widget._add_footer_summary,
            widget._refine_footer_summary,
        )
        seen = set()
        try:
            for _tag in self._drive_every_footer_state(widget):
                for lbl in labels:
                    if lbl.text().strip():
                        seen.add((lbl.text(), lbl.height(), lbl.width() or 320))
            assert seen, "no footer text was produced at all"
            for text, reserved, width in sorted(seen):
                twin = QLabel(text)
                twin.setWordWrap(True)
                twin.setStyleSheet("color: #999; font-size: 11px;")
                try:
                    needed = twin.heightForWidth(width)
                finally:
                    twin.setParent(None)
                assert needed <= reserved, (
                    "{!r} needs {}px at {}px wide but the footer reserves "
                    "{}px".format(text[:50], needed, width, reserved)
                )
        finally:
            widget.hide()

    @pytest.mark.parametrize("name", VISIBLE_PAGES)
    def test_the_footer_never_eats_the_viewport(self, widget, name):
        from napari_epicycle.annotator import advanced_features_widget as M

        footer = _slots(widget, name)["footer"]
        widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 40, 12: 38, 13: 20})
        h = footer.sizeHint().height()
        assert h <= M._MAX_FOOTER_H, (
            "{} footer is {}px; the contract is two buttons and one line "
            "(<= {}px)".format(name, h, M._MAX_FOOTER_H)
        )


# --------------------------------------------------------------------------- #
# I-13 -- a scroll area never eats a value
# --------------------------------------------------------------------------- #
class TestWheelOverAControl:
    def test_a_wheel_over_an_unfocused_spin_box_does_not_change_it(self, widget):
        from qtpy.QtCore import QPoint, QPointF, Qt
        from qtpy.QtGui import QWheelEvent
        from qtpy.QtWidgets import QApplication

        sb = widget.split_min_size_sb
        sb.clearFocus()
        before = sb.value()
        ev = QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
        )
        QApplication.sendEvent(sb, ev)
        assert sb.value() == before, (
            "the wheel silently changed a parameter the user was scrolling past"
        )

    def test_a_wheel_over_a_FOCUSED_spin_box_still_works(self, widget):
        from qtpy.QtCore import QPoint, QPointF, Qt
        from qtpy.QtGui import QWheelEvent
        from qtpy.QtWidgets import QApplication

        sb = widget.split_min_size_sb
        sb.setFocus(Qt.OtherFocusReason)
        if not sb.hasFocus():
            pytest.skip("offscreen platform will not give the spin box focus")
        before = sb.value()
        ev = QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
        )
        QApplication.sendEvent(sb, ev)
        assert sb.value() != before

    def test_the_guard_declines_when_there_is_no_scroll_area(self, qapp):
        """mode="hitl_2d" has no scroll area anywhere.  Returning True there
        would eat the wheel on all of its inputs and scroll nothing."""
        from qtpy.QtCore import QPoint, QPointF, Qt
        from qtpy.QtGui import QWheelEvent
        from qtpy.QtWidgets import QSpinBox

        from napari_epicycle.annotator import advanced_features_widget as M

        guard = M._WheelGuard()
        loose = QSpinBox()
        try:
            ev = QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
            )
            assert guard.eventFilter(loose, ev) is False, (
                "with no page to scroll, the wheel belongs to the control"
            )
        finally:
            loose.setParent(None)

    def test_a_hitl_2d_panel_keeps_its_wheel(self, qapp):
        from qtpy.QtCore import QPoint, QPointF, Qt
        from qtpy.QtGui import QWheelEvent
        from qtpy.QtWidgets import QApplication

        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        w2 = AdvancedFeaturesWidget(mode="hitl_2d")
        try:
            combo = w2.otsu_method_combo
            combo.clearFocus()
            before = combo.currentIndex()
            ev = QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
            )
            QApplication.sendEvent(combo, ev)
            assert combo.currentIndex() != before, (
                "the 2D panel has no scroll area; its inputs must keep the "
                "wheel"
            )
        finally:
            w2.setParent(None)

    def test_every_input_takes_focus_only_on_click_or_tab(self, widget):
        """StrongFocus (not WheelFocus) is half the fix: an input must not
        acquire focus just because the pointer scrolled over it."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

        for cls in (QSpinBox, QDoubleSpinBox, QComboBox):
            for w in widget.findChildren(cls):
                assert w.focusPolicy() == Qt.StrongFocus, (
                    "{} still takes wheel focus".format(type(w).__name__)
                )


# --------------------------------------------------------------------------- #
# I-14 -- every swap restores a sane viewport and focus
# --------------------------------------------------------------------------- #
class TestScrollAndFocusAcrossSwaps:
    def test_a_stage_change_puts_the_status_line_back_in_view(self, widget):
        s = _slots(widget, "Reshape")
        bar = s["scroll"].verticalScrollBar()
        bar.setValue(bar.maximum())
        widget.update_split_guided_ui(
            pieces=[11, 12], piece_sizes={11: 10, 12: 10})
        assert bar.value() == bar.minimum(), (
            "S5 appeared under a retained scroll position -- the user is "
            "looking at the wrong part of the panel"
        )

    def test_review_scrolls_the_RESULT_into_view_not_the_top(self, widget):
        """At the launcher's 340x350 the result starts below the fold, so
        scrolling to S1 puts the user's eyes on the status line while the
        thing to decide is off screen."""
        from qtpy.QtWidgets import QApplication

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        s = _slots(widget, "Reshape")
        widget.update_split_ui_state(preview_enabled=True)
        widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 4000, 12: 3800, 13: 2000},
            caption="unticked pieces stay part of cell 149")
        widget.update_split_ui_state(review_active=True)
        QApplication.processEvents()
        bar = s["scroll"].verticalScrollBar()
        if bar.maximum() == bar.minimum():
            widget.hide()
            pytest.skip("the whole page fits; nothing to scroll")
        top = widget._split_result_container.mapTo(
            s["scroll"].viewport(), widget._split_result_container.rect()
            .topLeft()).y()
        widget.hide()
        assert top >= 0, "the result is scrolled off the top"
        assert top < s["scroll"].viewport().height(), (
            "the result is below the fold after the swap that created it"
        )

    def test_a_non_review_swap_still_goes_back_to_the_status_line(self, widget):
        s = _slots(widget, "Reshape")
        bar = s["scroll"].verticalScrollBar()
        bar.setValue(bar.maximum())
        widget.update_split_ui_state(split_active=True, review_active=False)
        assert bar.value() == bar.minimum()

    def test_focus_lands_on_preview_not_on_accept(self, widget):
        """Qt reassigns focus when a focused widget is hidden, and Accept is
        the likely default button -- a held Return would commit.

        Preview is enabled first because that is the state the orchestrator
        pushes at review ("Preview again"); focus is never moved onto a
        disabled button.
        """
        widget.update_split_ui_state(preview_enabled=True)
        widget.update_split_guided_ui(
            pieces=[11, 12], piece_sizes={11: 10, 12: 10})
        assert widget.focusWidget() is widget.split_preview_btn

    def test_focus_is_not_moved_onto_a_disabled_preview(self, widget):
        widget.update_split_ui_state(preview_enabled=False)
        widget.update_split_guided_ui(
            pieces=[11, 12], piece_sizes={11: 10, 12: 10})
        assert widget.focusWidget() is not widget.split_accept_btn

    def test_focus_is_never_inside_a_hidden_container(self, widget):
        widget.update_split_guided_ui(busy=True)
        widget.update_split_guided_ui(busy=False)
        focused = widget.focusWidget()
        if focused is None:
            return
        node = focused
        while node is not None and node is not widget:
            assert node.isVisibleTo(widget) or not node.isHidden(), (
                "focus is parked inside a hidden container"
            )
            node = node.parentWidget()


# --------------------------------------------------------------------------- #
# Brick 2a -- the real-estate rule
#
# Measured: viewport 247 px at the launcher's 340x350 dock.  Two ceilings,
# because a page that fits vertically and overflows horizontally is silently
# CLIPPED -- the horizontal scrollbar is off.
# --------------------------------------------------------------------------- #
class TestS3FitsTheDock:
    def _shown(self, widget):
        from qtpy.QtWidgets import QApplication

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Reshape")
        QApplication.processEvents()
        return QApplication

    def _markers_found(self, widget):
        M = __import__(
            "napari_epicycle.annotator.advanced_features_widget",
            fromlist=["x"],
        )
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        widget.update_split_ui_state(
            split_active=True,
            marker_ids=[1, 2, 3, 4, 5],
            marker_stats={i: 1000 * i for i in range(1, 6)},
        )

    def _review(self, widget):
        M = __import__(
            "napari_epicycle.annotator.advanced_features_widget",
            fromlist=["x"],
        )
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        widget.update_split_ui_state(split_active=True)
        widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 4000, 12: 3800, 13: 2000},
            caption="unticked pieces stay part of cell 149",
            arbiter_message="some cuts had no membrane behind them",
            split_anyway_visible=True,
            engine_messages=["u-Segment3D is not installed"],
        )
        # `refine_further_enabled=True` is what a real guided review
        # pushes (Brick U4), and this walker never did -- so the two
        # ceiling tests below were measuring a page with the hand-off
        # button MISSING, and passing vacuously (audit H2).
        widget.update_split_ui_state(
            review_active=True, accept_enabled=True,
            refine_further_enabled=True,
            # Brick U3: a real guided review offers the wall-picture
            # tick too, for exactly the reason the line above exists --
            # a walker that does not push what the orchestrator pushes
            # measures a page the user never sees.
            walls_toggle_enabled=True,
        )

    #: The rule: the markers-found state fits two screens.
    #:
    #: "A screen" is the height the DOCK gives the page below its status
    #: furniture, which is the scroll viewport PLUS the pinned action
    #: strip -- pinning S4 (2026-09-19) moved 33 px from one of those
    #: buckets to the other without making the panel one pixel taller, so
    #: measuring against the shrunken viewport alone would have tightened
    #: a rule nobody decided to tighten.  `_screen_h` is the denominator.
    #:
    #: Measured 456 px against a 490 px ceiling after the pin (the
    #: content itself lost S4's 27 px).  34 px in hand.
    #:
    #: RAISED to 2.3 in Brick U1 (UX round 2.2).  The painting block --
    #: Markers: n, the distance heatmap, Add marker, Undo, and the one
    #: marker list -- is keyed on the SESSION now, not on the source, so
    #: it is in this state for the first time; it is ~83 px, and it is
    #: the owner-requested change, not drift.  Trimmed back what could
    #: be trimmed first: All/None moved onto the caption row, "Edit
    #: markers" is on screen only while there is a preview to go back
    #: from, the paint hint gives way to the list, and both captions
    #: elide rather than wrap.  Measured 539 px against a 559 px
    #: ceiling: 20 px in hand.
    #:
    #: RE-MEASURED, Brick U4 (2026-09-21): **540 px**, 19 px in hand.
    #: Dropping the grey line under the marker list took 14 px off this
    #: state too, and the hand-off button is not on screen in it -- the
    #: 1 px difference is the dropped line's spacing settling.
    #:
    #: UNCHANGED by Brick U3 (2026-09-21): still **540 px**.  Its one new
    #: control lives on the RESULT card's header row, and S5 is not on
    #: screen in this state at all.
    MARKERS_FOUND_SCREENS = 2.3

    #: The THIRD measured state (UX round 1.4): a preview on screen with
    #: the markers still open.  Both ticklists are up at once -- that is
    #: the point of the loop -- so it is the tallest state the panel has.
    #: The spec wrote 3.0 before anything was measured; the built page is
    #: 737 px against a 243 px screen, i.e. 3.03.  Recorded at 3.1 (753
    #: px, 16 px in hand) rather than deleting a control the round asked
    #: for.  S4 Preview is PINNED, so the one button the user has to
    #: reach is on screen at any of these heights -- which is the reason
    #: a three-screen state is tolerable at all.
    #:
    #: RE-MEASURED, Brick U2 (2026-09-20): **752 px against this 753 px
    #: ceiling -- 1 px in hand.**  The constant did NOT move; the page
    #: did.  Brick U2 added ONE control to S2, the outline tick, and it
    #: is 13 px (737 -> 752; the other 2 px are the row's spacing).  It
    #: is deliberately the only part of that brick visible during a
    #: session: its grey helper line is hidden with `_split_target_hint`
    #: -- shown, this state is 789 px and over -- while the tick itself
    #: stays up, because a session, painting markers in the side views,
    #: is exactly when you want to see through the object.
    #:
    #: So UX-round 1.4's "neither number is to be raised again without
    #: the owner" has become load-bearing rather than cautionary: the
    #: NEXT control added to this page does not fit, and the answer is
    #: the conversation 1.4 describes about what comes off the page --
    #: not a bigger number here.
    #:
    #: RE-MEASURED AGAIN, Brick U4 (2026-09-21), and the conversation
    #: above is the one that happened.  Two things:
    #:
    #: 1. This test was VACUOUS (audit H2).  `_review()` never pushed
    #:    `refine_further_enabled=True`, so every measurement since the
    #:    hand-off button existed was taken with it MISSING.  With it on
    #:    screen the state is **778 px** -- 25 px over the ceiling, and
    #:    it had been over since Brick U4 landed without anyone seeing.
    #: 2. The constant STILL did not move.  Two things came off the page
    #:    instead, as 1.4 says they must: the grey line under the marker
    #:    list ("Unticked markers are lifted off...", **-14 px**, its job
    #:    now done by the object itself since U2b) and the marker list's
    #:    own cap WHILE A PREVIEW IS ON SCREEN (84 -> 67 px, **-11 px**,
    #:    `_MARKER_LIST_REVIEW_MAX_H`; in review the list you act on is
    #:    the piece list, and touching a marker puts the preview away and
    #:    gives the markers their full height back).
    #:
    #: **778 -> 747 px against this 753 px ceiling: 6 px in hand.**
    #: The other two states moved with it: markers-found 539 -> 540 px
    #: (the marker list is unchanged outside review; the 1 px is the
    #: dropped line's spacing settling) and review-without-markers
    #: 457 px.
    #:
    #: RE-MEASURED, Brick U3 (2026-09-21): **749 px, 4 px in hand.**  The
    #: constant did not move and nothing came off the page, because the
    #: brick's one new control was placed so that it would not need to.
    #: "Show what the tool sees" rides the RESULT card's TITLE ROW: a
    #: row of its own is 15 px and there were 6, and the S2 object card
    #: -- the other candidate -- does not fit it sideways either (the
    #: two tick labels want 463 px of a 321 px card, measured).  On the
    #: header row it costs **2 px**: the checkbox is 13 px tall and the
    #: card title is 11.  markers-found is unchanged at 540 px (S5 is
    #: not on screen in it) and review-without-markers is 459 px.
    #:
    #: TICKED, the state is **763 px** -- 14 px more, which is the
    #: legend, the one line the user asked for by ticking the box, and
    #: nothing else.  That is 10 px past the ceiling, deliberately and
    #: only while the box is on; `test_reshape_walls_overlay.py`'s
    #: `TestTheHeightCeilingHolds` pins both halves of that sentence.
    REVIEW_MARKERS_SCREENS = 3.1

    def _screen_h(self, widget):
        """One screen of the Reshape page: the scroll viewport plus
        anything pinned between it and the footer."""
        slots = _slots(widget, "Reshape")
        h = slots["scroll"].viewport().height()
        strip = getattr(widget, "_split_s4_container", None)
        if strip is not None and strip.parentWidget() is slots["page"]:
            h += strip.height()
        return h

    def test_the_markers_found_state_is_at_most_two_screens(self, widget):
        app = self._shown(widget)
        try:
            self._markers_found(widget)
            app.processEvents()
            slots = _slots(widget, "Reshape")
            viewport = self._screen_h(widget)
            content = slots["content"].sizeHint().height()
            assert content <= self.MARKERS_FOUND_SCREENS * viewport, (
                "markers-found is {} px in a {} px screen -- {:.2f} screens "
                "of scrolling to see all of it".format(
                    content, viewport, content / max(viewport, 1))
            )
        finally:
            widget.hide()

    def _touching_found(self, widget, *, none_kept):
        """Markers from another run, some (or all) of them touching."""
        M = __import__(
            "napari_epicycle.annotator.advanced_features_widget",
            fromlist=["x"],
        )
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_RUN_LABELS)
        ids = [1, 2, 3, 4, 5]
        widget.update_split_ui_state(
            split_active=True, marker_ids=ids,
            marker_stats={i: 1000 * i for i in ids},
            marker_start_off={
                i: {"inside": 30, "object_voxels": 7200}
                for i in (ids if none_kept else [4, 5])},
        )

    @pytest.mark.parametrize("none_kept", [False, True])
    def test_the_touching_markers_state_is_within_the_same_ceiling(
            self, widget, none_kept):
        """The kept line and the none-kept line wrap longer than "Found 5
        markers"; both stay under the markers-found ceiling."""
        app = self._shown(widget)
        try:
            self._touching_found(widget, none_kept=none_kept)
            app.processEvents()
            text = widget.split_marker_count_label.text()
            assert text.startswith("None of the") is none_kept, text
            viewport = self._screen_h(widget)
            content = _slots(widget, "Reshape")["content"].sizeHint().height()
            print("[RESHAPE-HEIGHT] touching, none kept={}: {} px in {} px"
                  .format(none_kept, content, viewport))
            assert content <= self.MARKERS_FOUND_SCREENS * viewport, (
                content, viewport)
        finally:
            widget.hide()

    def test_the_review_state_is_at_most_two_screens(self, widget):
        app = self._shown(widget)
        try:
            self._review(widget)
            app.processEvents()
            slots = _slots(widget, "Reshape")
            viewport = self._screen_h(widget)
            content = slots["content"].sizeHint().height()
            assert content <= 2 * viewport, (
                "review is {} px in a {} px screen".format(
                    content, viewport)
            )
        finally:
            widget.hide()

    def test_the_review_with_the_left_out_line_is_within_two_screens(
            self, widget):
        """Preview with no Find puts one more grey line beside the cut."""
        from napari_epicycle.annotator import _reshape_text as T

        app = self._shown(widget)
        try:
            self._review(widget)
            widget.update_split_guided_ui(engine_messages=[
                T.MARKERS_LEFT_OUT.format(n=3, s="s"),
                "u-Segment3D is not installed"])
            app.processEvents()
            viewport = self._screen_h(widget)
            content = _slots(widget, "Reshape")["content"].sizeHint().height()
            print("[RESHAPE-HEIGHT] review + left-out line: {} px in {} px"
                  .format(content, viewport))
            assert content <= 2 * viewport, (content, viewport)
        finally:
            widget.hide()

    def _idle(self, widget):
        pass

    def _started(self, widget):
        widget.update_split_ui_state(split_active=True)

    def _advanced_open(self, widget):
        """The state a real-button shots run caught overflowing: opening
        Advanced put a horizontal scrollbar on the page and clipped the
        right edge -- Preview and the Sources line -- off it."""
        widget.split_advanced_btn.setChecked(True)

    def _everything_open(self, widget):
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_advanced_btn.setChecked(True)

    def _idle_sources_open(self, widget):
        """The driver's "idle" (its 12b): Sources > Markers open.

        Its section 2 opens them to reach the fifth channel combo and
        never closes them, which is also what a user does -- the
        disclosure is where you go BEFORE pressing Start.
        """
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)

    def _started_sources_open(self, widget):
        """The driver's "Start" (its 12d): a session, Sources > Markers
        open, and the image seed source picked, which is the widest the
        Markers card gets -- channel row, detector row, its two-line grey
        line, the prepare chip and the Find row."""
        M = __import__(
            "napari_epicycle.annotator.advanced_features_widget",
            fromlist=["x"],
        )
        widget.update_split_ui_state(split_active=True)
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)

    def _howto_open(self, widget):
        """The state H1 was found in: the How-to-cut card open, with the
        expected-count row's tick, spin and unit on one line."""
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_solve_btn.setChecked(True)

    def _walls_open(self, widget):
        """Walls open on a SHAPE wall -- the channel row and the prep chip
        are hidden, so this is the narrow case."""
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_walls_btn.setChecked(True)

    def _walls_open_image(self, widget):
        """The same card on an IMAGE wall, which reveals the channel row
        and the prepare chip.  The wide case."""
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.split_guided_summary_btn.setChecked(True)
        widget.split_walls_btn.setChecked(True)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)

    @pytest.mark.parametrize("state", ["idle", "started", "markers_found",
                                       "advanced_open", "everything_open",
                                       "howto_open", "walls_open",
                                       "walls_open_image"])
    def test_the_reshape_page_never_overflows_the_viewport(self, widget, state):
        """The horizontal scrollbar is OFF, so overflow is not scrolled to
        and not elided -- it is silently CLIPPED.  The ceiling is the
        VIEWPORT (329 px at the launcher's 340x350 dock), not the dock:
        the vertical scrollbar has already taken its share."""
        app = self._shown(widget)
        try:
            getattr(self, "_" + state)(widget)
            app.processEvents()
            slots = _slots(widget, "Reshape")
            viewport = slots["scroll"].viewport().width()
            width = slots["content"].minimumSizeHint().width()
            assert width <= viewport, (
                "Reshape/{} needs {} px in a {} px viewport -- the "
                "overflow is clipped, not scrolled".format(
                    state, width, viewport)
            )
        finally:
            widget.hide()

    #: The viewport the owner actually has.  The 340x350 dock the
    #: launcher opens leaves 245 px here; on the owner's machine, with
    #: the napari dock furniture around it, it is 209 (owner screenshot,
    #: 2026-09-18 -- "the Preview button is below the fold").  The widget
    #: height that reproduces it is 314.
    LIVE_VIEWPORT_H = 209
    LIVE_WIDGET_H = 314

    def _shown_small(self, widget):
        from qtpy.QtWidgets import QApplication

        widget.resize(340, self.LIVE_WIDGET_H)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Reshape")
        QApplication.processEvents()
        return QApplication

    @pytest.mark.parametrize("state", [
        "idle", "started",
        # The two the DRIVER measures, and the two that were failing
        # live at 280 px and 395 px against a 207 px viewport while the
        # two above passed at 167 px and 198 px.
        "idle_sources_open", "started_sources_open",
    ])
    def test_preview_is_above_the_fold_at_the_live_viewport(
            self, widget, state):
        """The one thing the user has to reach before anything happens.

        Scrolling to it is not the same as seeing it.  Measured against
        the PAGE rather than the scroll content, because that is the
        question -- is the button on screen -- and because S4 is not
        inside the scroll any more (see the test below for why).
        """
        app = self._shown_small(widget)
        try:
            getattr(self, "_" + state)(widget)
            app.processEvents()
            slots = _slots(widget, "Reshape")
            viewport = slots["scroll"].viewport().height()
            assert viewport <= self.LIVE_VIEWPORT_H + 40, (
                "the fixture no longer reproduces the live viewport: "
                "{} px".format(viewport)
            )
            page = slots["page"]
            btn = widget.split_preview_btn
            bottom = btn.mapTo(page, btn.rect().bottomLeft()).y()
            assert bottom <= page.height(), (
                "Reshape/{}: Preview's bottom edge is at {} px on a {} px "
                "page -- below the fold".format(state, bottom, page.height())
            )
        finally:
            widget.hide()

    def test_S4_is_PINNED_and_cannot_be_pushed_off_by_S3(self, widget):
        """The root cause, stated structurally.

        Preview used to sit below S3 INSIDE the scroll area, so it was
        pushed by whatever S3 was doing.  With one sub-section open on an
        image source, S3 alone measured 276-285 px live in a 207 px
        viewport -- bigger than the whole viewport -- so no amount of
        trimming descriptions or hints could have put Preview on screen.
        A fixed-height action that is needed in EVERY state does not
        belong in a scrolling region; it is pinned between the scroll and
        the footer now, and cannot be pushed by anything.
        """
        slots = _slots(widget, "Reshape")
        node, inside_scroll = widget._split_s4_container, False
        while node is not None:
            if node is slots["content"]:
                inside_scroll = True
                break
            node = node.parentWidget()
        assert not inside_scroll, (
            "S4 is back inside the scroll content -- S3 can push it off"
        )
        page_layout = slots["page"].layout()
        i_scroll = page_layout.indexOf(slots["scroll"])
        i_s4 = page_layout.indexOf(widget._split_s4_container)
        i_foot = page_layout.indexOf(slots["footer"])
        assert i_scroll < i_s4 < i_foot, (
            "S4 must sit between the scroll and the footer, in that order"
        )

    def test_the_pinned_strip_does_not_move_when_the_bar_appears(
            self, widget):
        """The footer's own lesson, applied one strip up: a pinned thing
        that changes height is not pinned.  The busy bar's slot is
        reserved, so starting a preview does not shift the button the
        user is about to look at."""
        app = self._shown_small(widget)
        try:
            idle_h = widget._split_s4_container.height()
            widget.update_split_guided_ui(busy=True)
            app.processEvents()
            assert widget._split_s4_container.height() == idle_h
            widget.update_split_guided_ui(busy=False)
            app.processEvents()
            assert widget._split_s4_container.height() == idle_h
        finally:
            widget.hide()

    def test_the_busy_bar_is_still_shown_and_hidden(self, widget):
        """Reserving the slot must not make the bar permanent."""
        app = self._shown_small(widget)
        try:
            widget.update_split_guided_ui(busy=True)
            app.processEvents()
            assert widget.split_busy_bar.isVisibleTo(widget) is True
            widget.update_split_guided_ui(busy=False)
            app.processEvents()
            assert widget.split_busy_bar.isVisibleTo(widget) is False
        finally:
            widget.hide()

    def test_the_expected_count_row_fits_beside_its_spin_box(self, widget):
        """H1, stated at the row rather than at the page.

        "I expect exactly this many cells" is 262 px of label; with the
        spin box beside it the row alone asked for 358 px in a 316 px
        viewport, and because the horizontal scrollbar is off the WHOLE
        page clipped -- Preview, Start and the spin box itself -- the
        moment How-to-cut was opened.  The words after the number are a
        separate label, so only "I expect exactly" is in the tick."""
        app = self._shown(widget)
        try:
            widget.split_guided_summary_btn.setChecked(True)
            widget.split_solve_btn.setChecked(True)
            app.processEvents()
            slots = _slots(widget, "Reshape")
            viewport = slots["scroll"].viewport().width()
            row = widget.split_target_count_cb.parentWidget()
            assert widget.split_target_count_cb.sizeHint().width() < 180, (
                "the tick alone is {} px".format(
                    widget.split_target_count_cb.sizeHint().width())
            )
            assert slots["content"].minimumSizeHint().width() <= viewport, (
                "How-to-cut open needs {} px in a {} px viewport".format(
                    slots["content"].minimumSizeHint().width(), viewport)
            )
            assert row is not None
        finally:
            widget.hide()

    def test_the_count_row_still_reads_as_one_sentence(self, widget):
        """Splitting the label must not lose the word it was carrying."""
        from qtpy.QtWidgets import QLabel

        assert widget.split_target_count_cb.text() == "I expect exactly"
        texts = [lb.text() for lb in widget.findChildren(QLabel)]
        assert "objects" in texts, (
            "the unit after the number is gone -- the row now reads "
            "'I expect exactly [2]'"
        )

    def test_the_painting_rows_wait_for_a_session(self, widget):
        """They are ALL disabled before Start -- Add Seed, Undo seed,
        Detect Seeds, Edit Seeds and the heatmap each need the cell held
        out of the volume.  On screen in idle they only say "do a thing
        you cannot do yet", in the 98 px Preview needs."""
        app = self._shown_small(widget)
        try:
            assert widget._split_paint_rows_container.isVisibleTo(
                widget) is False
            widget.update_split_ui_state(split_active=True)
            app.processEvents()
            assert widget._split_paint_rows_container.isVisibleTo(
                widget) is True
        finally:
            widget.hide()

    def test_the_one_marker_list_waits_until_it_has_markers(self, widget):
        """An empty list is 80 px that says nothing.

        `split_seed_list` was RETIRED in Brick U1 (UX round 2.2):
        `seed_ids=` fills `split_marker_list` now, so painted markers
        land in the same tickable list a Find fills and every source can
        be cherry-picked."""
        app = self._shown_small(widget)
        try:
            assert not hasattr(widget, "split_seed_list"), (
                "the second list is retired -- one list, five sources")
            widget.update_split_ui_state(split_active=True)
            app.processEvents()
            assert widget.split_marker_list.isVisibleTo(widget) is False
            widget.update_split_ui_state(seed_ids=[5, 6])
            app.processEvents()
            assert widget.split_marker_list.isVisibleTo(widget) is True
            assert widget.split_marker_list.count() == 2
        finally:
            widget.hide()

    def test_the_two_ticklists_are_never_CONFUSABLE(self, widget):
        """inv. 9 WITHDRAWN and replaced (UX round 1.4).

        They are on screen together now -- that is the point of the
        loop -- so what has to hold is that they can never be read as
        the same thing: a permanent caption on each card, and a
        visibly different row treatment (a gold left edge on the marker
        rows; the piece rows keep the plain chip)."""
        app = self._shown(widget)
        try:
            self._markers_found(widget)
            app.processEvents()
            assert widget.split_marker_list.isVisibleTo(widget) is True
            self._review(widget)
            app.processEvents()
            assert widget.split_piece_list.isVisibleTo(widget) is True
            assert widget.split_marker_list.isVisibleTo(widget) is True, (
                "the review must not take the markers away -- there "
                "would be nothing to loop back to")
            # ...and each card says which list it is, permanently.
            assert "before the cut" in widget.split_marker_caption.text()
            assert "after the cut" in widget.split_piece_card_caption.text()
            assert (widget.split_marker_list.styleSheet()
                    != widget.split_piece_list.styleSheet())
        finally:
            widget.hide()

    def test_the_review_state_with_markers_open_is_at_most_three_screens(
            self, widget):
        """The third measured state (UX round 1.4).

        Review with Markers open is taller than the two-screen ceiling
        by construction -- both lists are on screen.  S4 Preview is
        PINNED, so it is reachable regardless; what this bounds is how
        far the rest of it can grow."""
        app = self._shown(widget)
        try:
            self._markers_found(widget)
            self._review(widget)
            app.processEvents()
            slots = _slots(widget, "Reshape")
            viewport = self._screen_h(widget)
            content = slots["content"].sizeHint().height()
            assert content <= self.REVIEW_MARKERS_SCREENS * viewport, (
                "review+markers is {} px in a {} px screen -- {:.2f} "
                "screens.  There is a 6 px margin here, measured on an "
                "OFFSCREEN sizeHint (Brick U4): a miss on another "
                "machine, or after adding one control, is a conversation "
                "with the owner about what comes off the Reshape page -- "
                "not a bump to this constant, which UX-round 1.4 says "
                "may not be raised again without them.".format(
                    content, viewport, content / max(viewport, 1))
            )
            # ...and the two hard assertions that go with it.
            node, inside = widget._split_s4_container, False
            while node is not None:
                if node is slots["content"]:
                    inside = True
                    break
                node = node.parentWidget()
            assert not inside, "S4 must still be pinned in this state"
        finally:
            widget.hide()


class TestTheRefinePageAfterBrickB1:
    """.

    Refine joined `PAGES_WITH_A_PINNED_ACTION` in Brick B1, so it owes the
    same two things Reshape owes: a **measured** height ceiling for its
    tallest state, and a page that does not overflow the viewport
    sideways (the horizontal scrollbar is off, so overflow is clipped).

    It owes one thing Reshape does not: its strip carries a Stop button
    and a stage line as well as the bar, and a pinned thing that changes
    height is not pinned.  So its height is asserted CONSTANT across
    idle, working and staged -- which is new here; Reshape's `:943` is a
    WIDTH test and the first draft of the spec cited it wrongly.
    """

    #: The tallest state Brick B1 can reach: **Settle, with the prepare
    #: chevron open**.  Measured offscreen at the launcher's 340x350 dock:
    #: **694 px** of content against a 201 px viewport plus the 44 px
    #: pinned strip = 245 px, i.e. **2.83 screens**.  Recorded at 2.9
    #: (710 px, 16 px in hand).
    #:
    #: The spec names the page's eventual tallest state as **B2's** --
    #: Active contour, a 2D orientation, the sampling sub-knob showing and
    #: the prepare chevron open, drawn at roughly 750 px -- and calls its
    #: constant `REFINE_CONTOUR_2D_SCREENS`.  That state does not exist
    #: yet: B1's contour page is a placeholder with no Orientation and no
    #: sampler, and it measures 631 px.  So B1 records the state it
    #: actually has, and B2 adds the other constant beside it.
    #:
    #: The other four B1 states, for the record: idle/Smooth 484 px,
    #: Settle with the prepare chevron CLOSED 565 px, contour (snap to
    #: edges) 631 px, Settle STAGED with a result card and a warning
    #: 799 px.  The staged one is over this ceiling, deliberately and only
    #: while a result is on screen -- S4 is PINNED, so Accept, Discard and
    #: Preview are all reachable at any of these heights, which is the
    #: reason a three-screen state is tolerable at all.
    #:
    #: Advanced open adds a further 141 px (835 px).  That is a disclosure
    #: the user opened and can close, and the same is true of Reshape's.
    REFINE_SETTLE_PREP_SCREENS = 2.9

    def _shown(self, widget):
        from qtpy.QtWidgets import QApplication

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Refine")
        QApplication.processEvents()
        return QApplication

    def _screen_h(self, widget):
        slots = _slots(widget, "Refine")
        h = slots["scroll"].viewport().height()
        strip = getattr(widget, "_refine_s4_container", None)
        if strip is not None and strip.parentWidget() is slots["page"]:
            h += strip.height()
        return h

    def _settle_prep_open(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        widget.refine_method_combo.setCurrentIndex(
            REFINE_METHOD_INDEX["settle"])
        widget.refine_prep_btn.setChecked(True)
        # every control on: a patch with no probability map (B2 audit UX-M4)
        widget.set_refine_guided_grow_available(False)

    def test_the_tallest_B1_state_is_within_its_measured_ceiling(self, widget):
        app = self._shown(widget)
        try:
            self._settle_prep_open(widget)
            app.processEvents()
            slots = _slots(widget, "Refine")
            viewport = self._screen_h(widget)
            content = slots["content"].sizeHint().height()
            assert content <= self.REFINE_SETTLE_PREP_SCREENS * viewport, (
                "Settle with the prepare chevron open is {} px in a {} px "
                "screen -- {:.2f} screens.  This number was MEASURED; "
                "raising it is a conversation with the owner about what "
                "comes off the page.".format(
                    content, viewport, content / max(viewport, 1))
            )
        finally:
            widget.hide()

    def test_S4_is_pinned_outside_the_scroll(self, widget):
        app = self._shown(widget)
        try:
            slots = _slots(widget, "Refine")
            node, inside = widget._refine_s4_container, False
            while node is not None:
                if node is slots["content"]:
                    inside = True
                    break
                node = node.parentWidget()
            assert not inside, "S4 must be pinned, not scrolled"
            assert widget._refine_s4_container.parentWidget() is slots["page"]
        finally:
            widget.hide()

    def test_the_pinned_strip_is_the_same_height_in_every_state(self, widget):
        """NEW in B1.  All three slots -- the button row, the bar and the
        stage line -- are reserved at construction and merely shown and
        hidden inside that box, so the button the user has just clicked
        cannot jump under the cursor."""
        app = self._shown(widget)
        try:
            box = widget._refine_s4_container
            app.processEvents()
            h_idle = box.height()

            widget.update_smooth_ui_state(
                busy=True, stop_visible=True,
                stage_text="Settling the outline onto the wall...")
            app.processEvents()
            h_working = box.height()

            widget.update_smooth_ui_state(
                busy=False, stop_visible=False, stage_text="Stopping...")
            app.processEvents()
            h_stopping = box.height()

            widget.update_smooth_ui_state(
                stage_text="", accept_enabled=True, reject_enabled=True,
                result_note="From Settle: the outline moved on 2,140 voxels.",
                breadcrumb="Smooth -> Settle")
            app.processEvents()
            h_staged = box.height()

            assert h_idle == h_working == h_stopping == h_staged, (
                "the pinned strip moved: idle={} working={} stopping={} "
                "staged={}".format(h_idle, h_working, h_stopping, h_staged)
            )
            assert h_idle > 0
        finally:
            widget.hide()

    @pytest.mark.parametrize("state", ["idle", "settle", "settle_prep_open",
                                       "contour", "contour_brightness",
                                       "advanced_open"])
    def test_the_refine_page_never_overflows_the_viewport(self, widget, state):
        """The horizontal scrollbar is OFF, so overflow is not scrolled to
        and not elided -- it is silently CLIPPED."""
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        app = self._shown(widget)
        try:
            if state == "settle":
                widget.refine_method_combo.setCurrentIndex(
                    REFINE_METHOD_INDEX["settle"])
            elif state == "settle_prep_open":
                self._settle_prep_open(widget)
            elif state in ("contour", "contour_brightness"):
                widget.refine_method_combo.setCurrentIndex(
                    REFINE_METHOD_INDEX["contour"])
                widget.refine_contour_mode_combo.setCurrentIndex(
                    1 if state == "contour_brightness" else 0)
            elif state == "advanced_open":
                widget.refine_advanced_btn.setChecked(True)
            app.processEvents()
            slots = _slots(widget, "Refine")
            viewport = slots["scroll"].viewport().width()
            width = slots["content"].minimumSizeHint().width()
            assert width <= viewport, (
                "Refine/{} needs {} px in a {} px viewport -- the overflow "
                "is clipped, not scrolled".format(state, width, viewport)
            )
        finally:
            widget.hide()


class TestTheRefinePageAfterBrickB2:
    """The page's TALLEST state, which B2 built.

    **Active contour, Snap to edges, a plane orientation (XY) with the
    slice sampler showing, the prepare chevron open -- and every control
    on**: the push set below the gate so its LIVE line is the three-line
    amber "No effect at this setting" rather than the one-line neutral one
    (a ceiling measured on the short line would be vacuous).
    Measured offscreen at the launcher's 340x350 dock, 245 px per screen,
    after the B2 audit (every state with NO probability map, so a greyed
    Guided Grow is part of the measurement -- UX-M4):

    ==========================================  =======  =======
    state                                       content  screens
    ==========================================  =======  =======
    idle (Smooth)                                 381     1.56
    contour, snap, 3D, prepare open               675     2.76
    contour, snap, XY, prepare closed             610     2.49
    **contour, snap, XY, prepare open, push 0.2** **751** **3.07**
    contour, brightness, XY, prepare open         732     2.99
    Settle, prepare open                          707     2.89
    contour, snap, XY, prepare AND Advanced open  972     3.97
    ==========================================  =======  =======

    The last is over it deliberately -- Advanced is a disclosure the user
    opened -- as it is on Reshape.  S4 is PINNED, so Preview, Accept and
    Discard are reachable at every one of these heights.

    The ceiling is 3.12 (763 px), accepted by the owner on 2026-09-24
; the page measures 751 px under it -- still 751 after
    B3's help lines, which went onto the five short pages instead.  Not raised again without the owner: the next row
    added to this page comes off first.  (The prepare card's "Local
    contrast is not used..." sentence was that row; the 'Edge may move'
    grey line came off for it, and is the knob's tooltip now.)
    """

    REFINE_CONTOUR_2D_SCREENS = 3.12

    # The B1 class's two helpers, reused rather than inherited -- a subclass
    # would run every B1 test a second time under this name.
    _shown = TestTheRefinePageAfterBrickB1._shown
    _screen_h = TestTheRefinePageAfterBrickB1._screen_h

    def _contour_2d_every_control_on(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        widget.refine_method_combo.setCurrentIndex(
            REFINE_METHOD_INDEX["contour"])
        widget.refine_contour_mode_combo.setCurrentIndex(0)
        widget.refine_contour_axis_combo.setCurrentIndex(1)     # XY
        widget.refine_contour_slices_combo.setCurrentIndex(0)   # a share
        widget.refine_contour_balloon_sb.setValue(0.2)          # amber line
        widget.refine_prep_btn.setChecked(True)
        # every control on: a patch with no probability map (B2 audit UX-M4)
        widget.set_refine_guided_grow_available(False)

    def test_the_tallest_state_is_within_its_measured_ceiling(self, widget):
        app = self._shown(widget)
        try:
            self._contour_2d_every_control_on(widget)
            app.processEvents()
            # every control that state can show IS showing
            assert widget._refine_contour_sampling.isHidden() is False
            assert widget._refine_contour_share_row.isHidden() is False
            assert widget._refine_contour_push_box.isHidden() is False
            assert widget._refine_wall_container.isHidden() is False
            assert widget._refine_prep_container.isHidden() is False
            assert widget.refine_contour_push_line.text().startswith(
                "No effect")
            slots = _slots(widget, "Refine")
            viewport = self._screen_h(widget)
            content = slots["content"].sizeHint().height()
            print("[B2-HEIGHT] settle/contour -- contour 2D every control on: {} px in {} px "
                  "= {:.3f} screens".format(content, viewport,
                                            content / max(viewport, 1)))
            assert content <= self.REFINE_CONTOUR_2D_SCREENS * viewport, (
                "the tallest Refine state is {} px in a {} px screen -- "
                "{:.2f} screens.  MEASURED; raising it is a conversation "
                "with the owner about what comes off the page.".format(
                    content, viewport, content / max(viewport, 1)))
        finally:
            widget.hide()

    @pytest.mark.parametrize("state", ["contour_xy", "contour_xy_adv",
                                       "brightness_xy"])
    def test_the_B2_states_never_overflow_the_viewport(self, widget, state):
        app = self._shown(widget)
        try:
            self._contour_2d_every_control_on(widget)
            if state == "contour_xy_adv":
                widget.refine_advanced_btn.setChecked(True)
            if state == "brightness_xy":
                widget.refine_contour_mode_combo.setCurrentIndex(1)
            app.processEvents()
            slots = _slots(widget, "Refine")
            viewport = slots["scroll"].viewport().width()
            width = slots["content"].minimumSizeHint().width()
            assert width <= viewport, (
                "Refine/{} needs {} px in a {} px viewport".format(
                    state, width, viewport))
        finally:
            widget.hide()

    def test_the_stack_is_as_tall_as_the_page_it_shows(self, widget):
        """A plain QStackedWidget asks for its TALLEST page, so every
        method paid for the contour's rows; the Refine stack answers for
        the current page."""
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        stack = widget._refine_params_stack
        widget.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX["sdf"])
        assert stack.sizeHint().height() == (
            stack.currentWidget().sizeHint().height())
        # The Smooth page is the tallest page INSIDE the stack since B3 gave
        # every short page its grey lines (the contour's rows live in the
        # lower block under the shared card, so its in-stack page is only
        # Mode + its line).
        tall = stack.sizeHint().height()
        widget.refine_method_combo.setCurrentIndex(
            REFINE_METHOD_INDEX["fill_holes"])
        assert stack.sizeHint().height() < tall


class TestTheRefinePageWithRecipes:
    """/ invariant 6.

    The Recipes menu button rides the pinned strip's EXISTING row, so the
    strip stays at its reserved height in every recipe state, the page's
    effective minimum width does not move (the strip, Stop showing and a
    40-character recipe name on the stage line, still needs less than the
    scroll content already does), and neither ceiling changes -- measured
    with the TALLEST sentence the runner writes into S1, on both ceiling
    pages.
    """

    _shown = TestTheRefinePageAfterBrickB1._shown
    _screen_h = TestTheRefinePageAfterBrickB1._screen_h
    _settle_prep_open = TestTheRefinePageAfterBrickB1._settle_prep_open
    _contour_2d = TestTheRefinePageAfterBrickB2._contour_2d_every_control_on

    NAME = "W" * 40            # a 40-character name of the widest glyph

    def _runner_sentences(self):
        from packages.epicycle_annotator import refine_recipe as rr

        # No S1 sentence carries the recipe's name (UX audit M1); the
        # widest step word, the largest numbers.
        n, obj = 12, 99999
        fill = dict(n=n, k=n, obj=obj, word="Active contour",
                    steps=rr.plural(n, "step"), staged=rr.plural(n, "step"),
                    total=24, name=self.NAME, summary="x")
        out = []
        for attr in dir(rr):
            if attr.startswith("S1_") and isinstance(getattr(rr, attr),
                                                     str):
                out.append(getattr(rr, attr).format(**fill))
        for attr in ("HEADER_FRAGMENT", "HEADER_INTERP", "HEADER_BUSY",
                     "FILE_UNREADABLE", "SAVE_NOTHING", "SAVE_TOO_MANY",
                     "NO_WORKSPACE"):
            out.append(getattr(rr, attr))
        out.append(rr.SAVE_FLOOD.format(k=12))
        out.append(rr.HEADER_OTHER_OBJECT.format(staged=99999, obj=99998))
        return sorted(set(out), key=len)

    def _staged_recipe_state(self, widget):
        widget.update_smooth_ui_state(
            busy=True, stop_visible=True, stop_enabled=True,
            stage_text="12/12 \u00b7 Plane 44 of 60 \u2014 about 88 s left")

    def test_the_tallest_recipe_S1_fits_both_ceilings(self, widget):
        sentences = self._runner_sentences()
        for setup, ceiling, label in (
                (self._contour_2d,
                 TestTheRefinePageAfterBrickB2.REFINE_CONTOUR_2D_SCREENS,
                 "contour 2D"),
                (self._settle_prep_open,
                 TestTheRefinePageAfterBrickB1.REFINE_SETTLE_PREP_SCREENS,
                 "Settle prepare")):
            app = self._shown(widget)
            try:
                setup(widget)
                self._staged_recipe_state(widget)
                app.processEvents()
                slots = _slots(widget, "Refine")
                viewport = self._screen_h(widget)
                worst = (0, "")
                for text in sentences:
                    widget.update_smooth_ui_state(status_text=text)
                    app.processEvents()
                    h = slots["content"].sizeHint().height()
                    worst = max(worst, (h, text))
                print("[RECIPE-HEIGHT] {}: tallest recipe S1 {} px in {} px "
                      "= {:.3f} screens (ceiling {}) -- {!r}".format(
                          label, worst[0], viewport, worst[0] / viewport,
                          ceiling, worst[1][:60]))
                assert worst[0] <= ceiling * viewport, (label, worst)
            finally:
                widget.update_smooth_ui_state(
                    busy=False, stop_visible=False, stage_text="",
                    status_text="")
                widget.hide()

    def _realistic_failure_sentences(self, widget):
        """What the annotator ACTUALLY writes into S1 on the failure paths
        the UX audit measured over the ceilings (775 / 719 px, 410 px wide):
        a store error carrying a Windows path, a KeyError from the panel,
        a recipe that cannot be read.  Driven through the real handlers on
        the recipe fake, around THIS panel."""
        import numpy as np

        from packages.epicycle_annotator import refine_recipe as rr
        from tests.annotator import _recipe_fake as rf

        import napari_epicycle.annotator.curation_3d_widget as mod

        f = rf.Fake(rf.vol_with_notch(), widget)
        out = []
        real_labels = mod.Labels
        mod.Labels = rf._Layer          # the fake's selected layer
        try:
            self._drive_failures(f, widget, out, rr, np)
        finally:
            mod.Labels = real_labels
        return out

    def _drive_failures(self, f, widget, out, rr, np):
        widget.set_refine_step_params("sdf", {
            "sigma_z": 1.5, "sigma_xy": 1.0, "preserve_topology": True})
        f._preview_smooth()
        f.name_answer = (self.NAME, True)
        f._save_refine_recipe_cb = lambda name, obj: (
            "[WinError 5] Access is denied: 'C:\\Users\\someone\\"
            "Desktop\\EpiCycle-Local-Testing\\artifacts\\step3\\"
            "refine_chain_recipes.json'")
        f._on_recipe_save()
        out.append(widget.smooth_status_label.text())
        f._reject_smooth()
        good = rr.recipe_to_json(
            [rr.RecipeStep("fill_holes", {"axis": "3D"})], created="",
            spacing_known=True, spacing=())
        f.store = {self.NAME: good,
                   "broken": dict(good, steps=[{"method": "sdf",
                                                "params": {"x": 1}}])}
        real = widget.set_refine_step_params
        widget.set_refine_step_params = lambda m, p: (
            False, "the page could not be set (KeyError('refine_fh_axis_"
            "combo_that_moved_in_a_later_version'))")
        try:
            f._on_recipe_load(self.NAME)
            out.append(widget.smooth_status_label.text())
            f._on_recipe_run(self.NAME)
            np.asarray(f.vol())
            out.append(widget.smooth_status_label.text())
        finally:
            widget.set_refine_step_params = real
        f._on_recipe_run("broken")
        out.append(widget.smooth_status_label.text())
        return out

    def test_realistic_failure_sentences_fit_both_ceilings_and_the_width(
            self, widget):
        for setup, ceiling, label in (
                (self._contour_2d,
                 TestTheRefinePageAfterBrickB2.REFINE_CONTOUR_2D_SCREENS,
                 "contour 2D"),
                (self._settle_prep_open,
                 TestTheRefinePageAfterBrickB1.REFINE_SETTLE_PREP_SCREENS,
                 "Settle prepare")):
            app = self._shown(widget)
            try:
                texts = self._realistic_failure_sentences(widget)
                from packages.epicycle_annotator import refine_recipe as rr

                assert texts == [
                    rr.S1_SAVE_FAILED,
                    rr.S1_LOAD_FAILED.format(word="Fill Holes"),
                    rr.S1_NOT_SET_FIRST.format(n=1, word="Fill Holes"),
                    rr.S1_UNREADABLE], texts
                setup(widget)
                app.processEvents()
                slots = _slots(widget, "Refine")
                viewport = self._screen_h(widget)
                width_idle = slots["content"].minimumSizeHint().width()
                worst_h, worst_w = 0, 0
                for text in texts:
                    widget.update_smooth_ui_state(status_text=text)
                    app.processEvents()
                    worst_h = max(worst_h, slots["content"].sizeHint()
                                  .height())
                    worst_w = max(worst_w, slots["content"]
                                  .minimumSizeHint().width())
                print("[RECIPE-FAIL-HEIGHT] {}: {} px in {} px = {:.3f} "
                      "screens (ceiling {}); width {} px (idle {})".format(
                          label, worst_h, viewport, worst_h / viewport,
                          ceiling, worst_w, width_idle))
                assert worst_h <= ceiling * viewport, (label, worst_h)
                assert worst_w == width_idle, (label, worst_w, width_idle)
            finally:
                widget.update_smooth_ui_state(status_text="")
                widget.hide()

    def test_the_strip_reservation_holds_the_row_with_the_button(self,
                                                                 widget):
        app = self._shown(widget)
        try:
            strip = widget._refine_s4_container
            row = widget.refine_recipes_btn.parentWidget()
            assert row is widget.smooth_preview_btn.parentWidget()
            self._staged_recipe_state(widget)
            app.processEvents()
            lay = strip.layout()
            need = (row.sizeHint().height() + 2 * lay.spacing()
                    + widget.refine_busy_bar.height()
                    + widget.refine_stage_label.sizeHint().height())
            assert strip.height() >= need, (strip.height(), need)
            assert widget.refine_recipes_btn.sizeHint().height() <= (
                widget.smooth_preview_btn.sizeHint().height())
        finally:
            widget.hide()

    def test_the_strip_is_44_px_in_every_recipe_state(self, widget):
        app = self._shown(widget)
        try:
            box = widget._refine_s4_container
            heights = {"idle": box.height()}
            self._staged_recipe_state(widget)                  # running
            app.processEvents()
            heights["running"] = box.height()
            widget.update_smooth_ui_state(stop_enabled=False,
                                          stage_text="3/3 \u00b7 Stopping...")
            app.processEvents()
            heights["stopping"] = box.height()
            widget.update_smooth_ui_state(
                busy=False, stop_visible=False,
                stage_text="Stopped at 2/3 \u00b7 Fill Holes \u2014 "
                           "'{}'".format(self.NAME))
            app.processEvents()
            heights["ended"] = box.height()
            assert set(heights.values()) == {44}, heights
        finally:
            widget.hide()

    def test_the_page_minimum_width_is_unchanged_with_a_40_char_name(
            self, widget):
        from packages.epicycle_annotator import refine_recipe as rr

        app = self._shown(widget)
        try:
            slots = _slots(widget, "Refine")
            strip = widget._refine_s4_container
            content_idle = slots["content"].minimumSizeHint().width()
            widget.update_smooth_ui_state(
                busy=True, stop_visible=True, stop_enabled=True,
                stage_text=rr.STAGE_STOPPED.format(
                    name=self.NAME, k=12, n=12, what=rr.STAGE_EMPTY),
                status_text=rr.S1_RUNNING.format(steps="12 steps",
                                                 obj=99999))
            app.processEvents()
            content_run = slots["content"].minimumSizeHint().width()
            strip_run = strip.minimumSizeHint().width()
            viewport = slots["scroll"].viewport().width()
            print("[RECIPE-WIDTH] content {} -> {} px, strip {} px, "
                  "viewport {} px".format(content_idle, content_run,
                                          strip_run, viewport))
            assert content_run == content_idle
            # the page needs what the content needs; the strip, with Stop
            # AND the button showing, needs less
            assert strip_run <= content_run
            assert content_run <= viewport
        finally:
            widget.hide()


class TestTheAddPageAfterRebuild:
    """The candidates states, measured through the
    REAL page (its switching parts are hidden widgets, never a stack), at
    the launcher's 340x350 dock, 245 px per screen (scroll viewport + the
    pinned strip), with EVERY optional control showing.

    **Measured at the width the content really gets** (B2 fix pass U-H3):
    the first fixture read ``content.sizeHint().height()``, i.e. the height
    at the content's size-hint WIDTH (507 px), not the viewport's -- and
    every one of these states overflows the viewport, so the vertical
    scrollbar is up and the content is 316 px wide, where every sentence
    wraps to more lines.  ``_content_h`` is the layout's height-for-width
    at ``_viewport_w`` (the viewport minus the scrollbar when Qt has not
    shown it yet).  A helper that omits a control measures a state the app
    never shows, so every state pushes: S1 at its longest
    reachable sentence, the counts note with every part non-zero (at its
    cap), the walls legend (or the grey reason on Nuclei), a 12-row list,
    the strip busy with Stop and a stage line.

    MEASURED -- see ``MEASURED`` below for the numbers and ``TAKEN_OFF``
    for what came off the page, order.  The ceiling is 3.1
    (759.5 px), the owner's; never raised without the owner.
    """

    ADD_CANDIDATES_SCREENS = 3.1
    STATES = ("7a", "7a+", "7b", "7c", "7d")

    #: Measured 2026-09-27 (offscreen, 316 px wide, 245 px screen), after
    #: the width fixes (U-H1) and after the Add lists took the panel's
    #: 12 px swatch (a row had been 32 px at Qt's default icon size, not
    #: the mockup's 19 -- a bug, not a take-off):
    #:
    #:   state  5 rows          step 1: 4 rows   final (notes elided to 2
    #:                                            lines, mockup parity)
    #:   7a     649 px  2.65    627 px  2.56     628 px  2.56
    #:   7a+    709 px  2.89    687 px  2.80     688 px  2.81
    #:   7b     707 px  2.89    685 px  2.80     681 px  2.78
    #:   7c     781 px  3.19    759 px  3.10     755 px  3.08  <- tallest
    #:   7d     733 px  2.99    711 px  2.90     707 px  2.89
    #:
    #: Width (content minimum, final): 312 px in the 316 px viewport in
    #: every state (393 px -- 631 with four prepare ticks -- before U-H1).
    #: The old fixture read 617 / 663 / 676 px at the 507 px size-hint
    #: width; the auditor's 329 px numbers were 671 / 731 / 729 / 778 /
    #: 745 with 32 px rows.
    #:
    #: **Live-shot fix pass (2026-09-27, V9): re-measured at 312 px.**  The
    #: themed live run showed the real viewport is 312 px wide, not 316
    #: (7d measured 751 px live, B03 765).  LIVE_VIEWPORT_W is now the
    #: measuring width.  With F1-F9 and V1-V8 in (the longer voxel-size
    #: sentence, the 26 px header chevron, S / M / L at their hints, the
    #: strip always drawn -- its height was already fixed, so V2 adds
    #: nothing inside the scroll area), and the 7a fixture now PICKING the
    #: nuclei channel (blank, its help is the short ADD_AUTO_NEEDS_CHANNEL):
    #:
    #:   state  start @312      step 2 @312      step 3 @312 (final)
    #:   7a     640 px  2.61    640 px  2.61     629 px  2.57
    #:   7a+    700 px  2.86    700 px  2.86     689 px  2.81
    #:   7b     693 px  2.83    669 px  2.73     658 px  2.69
    #:   7c     767 px  3.13    767 px  3.13     756 px  3.09  <- tallest
    #:   7d     719 px  2.93    719 px  2.93     708 px  2.89
    #:
    #: 7c was over 759.5, so next items came off in order:
    #: (2) REFINE_NO_CLAHE_SENTENCE off the prepare sub-chevron into the
    #: guide (the sub-chevron already opened closed): 7b 693 -> 669, 7c
    #: unchanged (its Walls are closed); (3) the counts note caps at ONE
    #: line (ADD_NOTES_LINES 2 -> 1): 7c 767 -> 756, under 759.5.  Step (4)
    #: was not needed; 3.1 was not raised.  Content minimum width 295 px.
    #:
    #: **Follow-up brick A1:** ADD_S1_NO_VOXEL_SIZE
    #: retired, so 7c's S1 is the longest pre-result sentence left
    #: (over_limit_line): 7c 756 -> 744 px (3.04 screens); the others
    #: unchanged.  The amber default-spacing note is never shown over
    #: candidates; its own state 7e measures 619 px (2.53 screens).
    #:
    #: **A3:** the outline row adds 16 px to every candidates
    #: state: 7a 645 · 7a+ 705 · 7b 674 · 7c 760 · 7d 724 -- 7c 0.5 px
    #: over 759.5.  Step (4) is DEAD (ADD_HELP_CLICKS wraps to three lines
    #: with or without its last sentence at 312 px: 0 px saved), replaced
    #: by (4'): ADD_AMBER_MEMBRANE_ONLY hidden over candidates and REVIEW
    #: -- 7c 760 -> 675 px.  7d is now the tallest (2.96 screens).
    #:
    #: **A4:** "Accept and refine" does not fit the footer row
    #: (381 px; 319 at 3 px padding) and as a second footer line it cut the
    #: viewport 245 -> 220 px (7a+ 705 and 7d 724 over 682).  It takes the
    #: footer summary's fixed 31 px slot while candidates are up, so the
    #: footer stays 66 px and every number above is unchanged.
    #:
    #: **A5:** the click-marker list is hidden over candidates
    #: and REVIEW, so 7a-7e are unchanged (re-measured: 645 / 705 / 674 /
    #: 675 / 724 / 619).  New pre-result state **7f** = 7e + 12 markers
    #: (the widest rows, "Object 655xx" at 1234 / 1234 / 1234) with the
    #: list at ADD_MARKER_LIST_ROWS = 3 rows: 689 px busy -- **which was
    #: the wrong number** (the A5 UX audit, C-MED): the fixture pushed
    #: ``busy=True``, and busy HIDES ``_add_box_help`` (ADD_HELP_DRAG, 50
    #: px), so it measured a state shorter than the idle page the user
    #: looks at while placing markers.  Re-measured (fix pass, 2026-09-27)
    #: as the max of busy and idle -- the idle page is the taller:
    #:
    #:   state        busy     idle (the number that counts)
    #:   7e           619 px   669 px  2.73 screens
    #:   7f, 3 rows   689 px   739 px  3.02 screens
    #:   7f, 4 rows   711 px   761 px  OVER 759.5 -- the grill's 761 WAS
    #:                                  right; 3 rows is the only fit
    #:
    #: **Do not raise ADD_MARKER_LIST_ROWS**: a fourth row is 1.5 px over.
    #: The Add page's scroll area takes no focus (M3).
    #:
    #: **A-M2 (the A1-A4 audit):** with a box the spacing note is the
    #: shorter ADD_NOTE_DEFAULT_SPACING_BOXED; 7e / 7f measure with it (the
    #: numbers above).
    #:
    #: **D-1 (the owner's live test):** with "Treat dark space as outside"
    #: OFF, the amber ADD_NOTE_DARK_OFF_LEAK (71 px, four lines) stands
    #: under S1 too.  STACKED with the spacing note, 7f idle measured
    #: **812 px** -- over.  So the dark-off note wins S1's slot and the
    #: spacing note moves to the Advanced help slot (under ADD_DARK_OFF):
    #: state **7f+dark** = 7f idle, the dark-off note, no spacing note on S1:
    #: **751 px** (3.07 screens).  3.1 was not raised.
    #: (7d re-measured 716 after Look for nuclei came in and the threshold
    #: help went off over candidates, then 728 when the Nuclei help named
    #: Inside each cell (three lines) -- see DERIVE_STATES below)
    #: Every state here is PINNED to its number (the page fix pass, L9): a
    #: change that moves one says so here, with the new measurement.
    #: (7b 674 -> 703, 2026-11-12: the Walls card gained its Tissue outline
    #: row, 29 px; its help line is hidden over candidates and Show the
    #: tissue outline needs an outline, so the row alone shows in 7b)
    #: (2026-11-16: a result with nothing ticked shows a line that names
    #: All.  The Cells and Anywhere states read it without a two-nuclei
    #: tail and do not move; 7d reads it with the tail -- 125 characters,
    #: one more S1 line -- because its other test lists the Inside-only
    #: lines: 7d 728 -> 740)
    #: (7c 675 -> 677, 7e 669 -> 671, 7f 739 -> 741, 7f+dark 751 -> 753,
    #: 2026-11-16: the click markers' first row now holds Place markers, the
    #: word "Ball" and its radius spin box, whose height is 2 px over the
    #: button's; the placed count left that row; 7f+dark stays under 759.5)
    #: (7b 703 -> 718, 2026-11-24: the Tissue outline defaults to the
    #: shrink-wrap on a box too, so the Walls card shows Show the tissue
    #: outline under the row: 15 px; still under 7f+dark)
    MEASURED = {"7a": 645, "7a+": 705, "7b": 718, "7c": 677, "7d": 740,
                "7e": 671, "7f": 741, "7f+dark": 753}
    LIVE_VIEWPORT_W = 312

    #: What came off, order, re-measured after each:
    #: (1) ADD_LIST_ROWS 5 -> 4: 7c 781 -> 759 px (B2 fix pass, at 316);
    #: (2) the no-local-contrast sentence to the guide and (3) the counts
    #: note at one line (live-shot fix pass, at 312).  Step 4 not needed.
    TAKEN_OFF = ("ADD_LIST_ROWS 5 -> 4",
                 "REFINE_NO_CLAHE_SENTENCE -> the guide",
                 "ADD_NOTES_LINES 2 -> 1",
                 "ADD_AMBER_MEMBRANE_ONLY off over candidates / REVIEW")

    def _state(self, widget, state):
        from qtpy.QtWidgets import QApplication

        from packages.epicycle_annotator import add_preview as AP

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Add")
        QApplication.processEvents()
        single = state == "7c"
        widget.refresh_add_image_combo(["nuc"] if single else ["nuc", "mem"])
        # 7c is the membrane-only patch: BOTH automatic sources grey (no
        # other run, one channel), which is when the amber line shows
        items = [
            ((AP.ADD_ITEM_NO_RUN, False, ("labels", None)) if single else
             (AP.ADD_ITEM_RUN.format(name="a-rather-long-run-name-here-xyz")
              + AP.ADD_ITEM_RAW_SUFFIX, True, ("labels", "r"))),
            (AP.ADD_ITEM_NUCLEI_GREY if single else AP.ADD_ITEM_NUCLEI,
             not single, ("nuclei_masks", None)),
            (AP.ADD_ITEM_CLICKS, True, ("paint", None)),
        ]
        widget.set_add_preset("nuclei" if state == "7d" else "cells")
        widget.update_add_ui_state(marker_sources=items, box_active=True,
                                   single_channel=single,
                                   voxel_size_known=True)
        if state in ("7a", "7a+"):
            widget.add_marker_source_combo.setCurrentIndex(1)
            # a picked nuclei channel: blank, the help slot reads the SHORT
            # ADD_AUTO_NEEDS_CHANNEL (live-shot F9) -- not the state measured
            widget.add_nuclei_channel_combo.setCurrentIndex(0)
            widget.add_markers_btn.setChecked(True)
            widget.update_add_ui_state(
                nuclei_auto=float("nan") if state == "7a" else 0.214)
        elif state == "7b":
            widget.add_walls_btn.setChecked(True)
            widget.add_wall_prep_btn.setChecked(True)
            for key in ("zcorrect", "bgsub", "lift", "denoise"):
                getattr(widget, "add_prep_{}_cb".format(key)).setChecked(True)
        elif state == "7c":
            widget.add_marker_source_combo.setCurrentIndex(2)
            widget.add_markers_btn.setChecked(True)
            widget.update_add_ui_state(click_count=7)
        else:
            import numpy as np

            widget.update_add_ui_state(histogram_data={
                "values": list(np.linspace(0.0, 1.0, 500)), "n_bins": 32})
        ids = list(range(400, 412))
        widget.update_add_ui_state(candidate_small_ids={401, 402})
        widget.update_add_ui_state(
            candidate_ids=ids, candidate_sizes={i: 1234567 for i in ids},
            candidate_edge_ids={400, 401})
        counts = {k: 12 for k, _m, _o in AP._COUNT_PARTS}
        notes = [("counts", AP.counts_note(counts), False)]
        if state == "7d":
            widget.update_add_ui_state(
                notes=notes, walls_toggle_enabled=False,
                walls_toggle_reason=AP.WALLS_TOGGLE_NO_WALLS)
        else:
            widget.update_add_ui_state(notes=notes, walls_toggle_enabled=True,
                                       walls_toggle_reason="")
            widget.add_show_walls_cb.setChecked(True)
        # 7d is pinned to one number here and among the derive states,
        # whose 7d list carries the Inside-only lines (the two-nuclei
        # tail): it reads them here too
        widget.update_add_ui_state(status_text=self._longest_s1(
                                       single, two_nuclei=(state == "7d")),
                                   busy=True, stop_enabled=True,
                                   stage_text="Filling the empty space...")
        # offscreen Qt settles the scroll area a pass or two late, and the
        # first panel built in a process can come up wider than the dock
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        return QApplication

    @staticmethod
    def _longest_s1(single, two_nuclei=False):
        """The longest S1 sentence the state can reach: over candidates,
        the longest of the sentences that can stand above a result; the
        single-channel state (7c) at the longest sentence a Cells page on
        the clicks source can show before a result -- it was
        ADD_S1_NO_VOXEL_SIZE until retired it.  ``two_nuclei`` adds the
        nothing-ticked line with the two-nuclei tail, which only Nuclei ▸
        Inside each cell can show."""
        from packages.epicycle_annotator import add_preview as AP

        if single:
            return max((AP.over_limit_line((200, 200, 200)),
                        AP.ADD_S1_NO_MARKERS, AP.ADD_S1_NO_MARKERS_ON,
                        AP.ADD_S1_PICK_WALL_CHANNEL,
                        AP.ADD_S1_CLICK_ON_KEPT.format(id=123456),
                        AP.ADD_S1_CLICKS_DROPPED.format(n=12),
                        AP.ADD_S1_WHOLE_MARKERS_CLEARED,
                        AP.ADD_S1_BUSY_REFUSED, AP.ADD_S1_PARKED_REFUSED,
                        AP.ADD_S1_FAILED,
                        AP.ADD_S1_ALL_FILTERED, AP.ADD_S1_INPUTS_CHANGED,
                        # Update as I click: a Cells whole patch or box not
                        # run as you click, a Cells box or patch over the
                        # live limit, and an Accept waiting that ended
                        AP.ADD_S1_LIVE_OVER_GATE_PATCH,
                        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX,
                        AP.ADD_S1_LIVE_OVER_GATE_BOX.format(
                            n=AP.voxel_limit_words(
                                AP.ADD_LIVE_CELLS_MAX_VOXELS)),
                        AP.ADD_S1_LIVE_OVER_GATE_PATCH_SIZE.format(
                            n=AP.voxel_limit_words(
                                AP.ADD_LIVE_CELLS_MAX_VOXELS)),
                        # ... and a click waiting for the click fill
                        AP.ADD_S1_LIVE_PREPARING,
                        AP.ADD_S1_ACCEPT_WAIT_OFF,
                        AP.ADD_S1_CLEAR_WAIT_LIVE),
                       key=len)
        return max((AP.ADD_S1_TAKEN, AP.ADD_S1_FAILED, AP.ADD_S1_STOPPED,
                    AP.ADD_S1_INPUTS_CHANGED, AP.ADD_S1_BUSY_REFUSED,
                    AP.ADD_S1_PARKED_REFUSED, AP.ADD_S1_TICK_ONE,
                    # Update as I click: Accept waiting over a result
                    AP.ADD_S1_ACCEPT_WAITING, AP.ADD_S1_ACCEPT_WAIT_CLICK,
                    AP.ADD_S1_ACCEPT_WAIT_OFF,
                    AP.found_line(12, k_edge=2, k_small=2, k_both=1),
                    # a Shift+click refused over a result (the click
                    # markers)
                    AP.ADD_S1_NOT_CONNECTED.format(id=65520),
                    # nothing ticked: the line names All -- a result of
                    # hundreds of objects (a whole patch), no two-nuclei
                    # tail (a Cells or Anywhere result has none)
                    AP.found_none_ticked_line(999, AP.ADD_WHY_BOTH))
                   + ((AP.found_none_ticked_line(12, AP.ADD_WHY_BOTH, 12),)
                      if two_nuclei else ()),
                   key=len)

    def _screen_h(self, widget):
        s = _slots(widget, "Add")
        return s["scroll"].viewport().height() + widget._add_s4_container.height()

    def _viewport_w(self, widget):
        """The width the content gets with the vertical scrollbar up --
        which it is in every state measured here (each overflows).  From
        the scroll area's own width, because offscreen Qt shows the bar a
        layout pass late and ``viewport().width()`` is then 329, not 316."""
        from qtpy.QtWidgets import QStyle

        scroll = _slots(widget, "Add")["scroll"]
        bar = scroll.verticalScrollBar()
        extent = max(bar.sizeHint().width(),
                     scroll.style().pixelMetric(QStyle.PM_ScrollBarExtent))
        # at the launcher's 340 px dock -- the first panel built in a
        # process can come up a few px wider than it was asked to be
        over = max(0, widget.width() - 340)
        computed = scroll.width() - over - 2 * scroll.frameWidth() - extent
        # the live viewport is 312 px (live-shot fix V9), 4 px narrower
        # than offscreen Qt's arithmetic gives
        return min(computed, self.LIVE_VIEWPORT_W)

    def _content_h(self, widget):
        content = _slots(widget, "Add")["content"]
        lay = content.layout()
        vw = self._viewport_w(widget)
        if lay.hasHeightForWidth():
            return lay.totalHeightForWidth(vw)
        return content.sizeHint().height()

    @pytest.mark.parametrize("state", STATES)
    def test_the_candidates_states_are_within_the_ceiling(self, widget, state):
        app = self._state(widget, state)
        try:
            # every optional control that state can show IS showing
            assert not widget._add_result_container.isHidden()
            assert not widget.add_notes_label.isHidden()
            assert not widget.add_walls_legend.isHidden()
            assert not widget.add_contour_cb.isHidden()      # A3
            assert not widget.add_stop_btn.isHidden()
            assert widget.add_candidate_list.count() == 12
            if state == "7c":
                # the take-off that replaced step (4):
                # the membrane-only amber is a pre-result instruction, off
                # over candidates -- it is measured in 7e instead
                assert widget._add_membrane_only_label.isHidden()
                assert not widget._add_click_rows.isHidden()
            if state == "7b":
                assert not widget._add_wall_prep_container.isHidden()
            viewport = self._screen_h(widget)
            content = self._content_h(widget)
            print("[ADD-HEIGHT] {}: {} px at {} px wide in {} px = {:.3f} "
                  "screens".format(state, content, self._viewport_w(widget),
                                   viewport, content / max(viewport, 1)))
            assert content <= self.ADD_CANDIDATES_SCREENS * viewport, (
                "Add {} is {} px in a {} px screen -- {:.2f} screens; trim the "
                "content, never raise the 3.1-screen ceiling".format(
                    state, content, viewport, content / max(viewport, 1)))
            assert content == self.MEASURED[state], (
                "Add {} moved: {} px, recorded {} -- record the new number "
                "in MEASURED with the reason".format(
                    state, content, self.MEASURED[state]))
            app.processEvents()
        finally:
            widget.hide()

    @pytest.mark.parametrize("state", STATES)
    def test_the_candidates_states_never_overflow_the_viewport(
            self, widget, state):
        """The horizontal scrollbar is off, so overflow is CLIPPED, not
        scrolled (B2 fix pass U-H1: 393 px of content in a 329 px
        viewport clipped "Show what the tool sees", Clear and the prepare
        chevron)."""
        app = self._state(widget, state)
        try:
            width = _slots(widget, "Add")["content"].minimumSizeHint().width()
            viewport = self._viewport_w(widget)
            print("[ADD-WIDTH] {}: {} px in a {} px viewport".format(
                state, width, viewport))
            assert width <= viewport, (
                "Add/{} needs {} px in a {} px viewport -- the overflow is "
                "clipped, not scrolled".format(state, width, viewport))
            app.processEvents()
        finally:
            widget.hide()

    def test_the_default_spacing_note_state_is_within_the_ceiling(
            self, widget):
        """State 7e: the amber default-spacing note
        is shown only BEFORE a result, so it is measured there -- Cells on a
        single-channel patch, clicks source, Markers open, the note, S1 at
        its longest pre-result sentence, the strip busy."""
        from qtpy.QtWidgets import QApplication

        from packages.epicycle_annotator import add_preview as AP

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Add")
        widget.refresh_add_image_combo(["nuc"])
        widget.set_add_preset("cells")
        widget.update_add_ui_state(marker_sources=[
            (AP.ADD_ITEM_NO_RUN, False, ("labels", None)),
            (AP.ADD_ITEM_NUCLEI_GREY, False, ("nuclei_masks", None)),
            (AP.ADD_ITEM_CLICKS, True, ("paint", None))],
            box_active=True, single_channel=True, default_spacing=True)
        widget.add_marker_source_combo.setCurrentIndex(2)
        widget.add_markers_btn.setChecked(True)
        widget.update_add_ui_state(
            click_count=7, candidate_ids=[], candidate_sizes={},
            candidate_edge_ids=set(), status_text=self._longest_s1(True),
            busy=True, stop_enabled=True, stage_text="Filling...")
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        try:
            assert not widget.add_spacing_note.isHidden()
            assert widget.add_spacing_note.text() == (
                AP.ADD_NOTE_DEFAULT_SPACING_BOXED)            # A-M2
            assert not widget._add_membrane_only_label.isHidden()
            assert widget._add_result_container.isHidden()
            viewport = self._screen_h(widget)
            busy, idle = self._busy_and_idle(widget)
            content = max(busy, idle)
            print("[ADD-HEIGHT] 7e: busy {} / idle {} px in {} px = {:.3f} "
                  "screens".format(busy, idle, viewport,
                                   content / max(viewport, 1)))
            assert content <= self.ADD_CANDIDATES_SCREENS * viewport
            assert content == self.MEASURED["7e"], content
            width = _slots(widget, "Add")["content"].minimumSizeHint().width()
            assert width <= self._viewport_w(widget)
        finally:
            widget.update_add_ui_state(default_spacing=False, busy=False)
            widget.hide()

    def _state_7f(self, widget, rows=None):
        """State 7f = 7e + 12 click markers: pre-result
        Cells on a single-channel patch, clicks source, Markers open, the
        amber default-spacing note, S1 at its longest pre-result sentence,
        the strip busy -- and the marker list at ADD_MARKER_LIST_ROWS rows
        (``rows`` overrides the cap, to measure the rejected 4)."""
        from qtpy.QtWidgets import QApplication

        from packages.epicycle_annotator import add_preview as AP

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Add")
        widget.refresh_add_image_combo(["nuc"])
        widget.set_add_preset("cells")
        widget.update_add_ui_state(marker_sources=[
            (AP.ADD_ITEM_NO_RUN, False, ("labels", None)),
            (AP.ADD_ITEM_NUCLEI_GREY, False, ("nuclei_masks", None)),
            (AP.ADD_ITEM_CLICKS, True, ("paint", None))],
            box_active=True, single_channel=True, default_spacing=True)
        widget.add_marker_source_combo.setCurrentIndex(2)
        widget.add_markers_btn.setChecked(True)
        markers = [(65520 + i, (1234, 1234, 1234), i % 3 != 1)
                   for i in range(12)]
        widget.update_add_ui_state(
            click_count=12, click_markers=markers, candidate_ids=[],
            candidate_sizes={}, candidate_edge_ids=set(),
            status_text=self._longest_s1(True),
            busy=True, stop_enabled=True, stage_text="Filling...")
        if rows is not None:
            widget._cap_add_list(widget.add_click_list, rows=rows)
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        return QApplication

    def _undo_7f(self, widget):
        widget.add_dark_outside_cb.setChecked(True)
        widget.update_add_ui_state(default_spacing=False, busy=False,
                                   click_count=0, click_markers=[])
        widget.hide()

    def _busy_and_idle(self, widget):
        """``(busy, idle)`` content heights (C-MED of the A5 UX audit):
        busy HIDES the box's drag help, so a busy-only measure is short by
        its 50 px.  Leaves the page idle."""
        from qtpy.QtWidgets import QApplication

        widget.update_add_ui_state(busy=True, stop_enabled=True,
                                   stage_text="Filling...")
        for _ in range(3):
            QApplication.processEvents()
        assert widget._add_box_help.isHidden(), "busy hides the drag help"
        busy = self._content_h(widget)
        widget.update_add_ui_state(busy=False)
        for _ in range(3):
            QApplication.processEvents()
        assert not widget._add_box_help.isHidden(), "idle shows it"
        return busy, self._content_h(widget)

    def test_the_marker_list_state_is_within_the_ceiling(self, widget):
        """State 7f: the marker list is shown only BEFORE
        a result, so it is measured on top of 7e -- 12 markers, the list
        capped at 3 rows (4 rows measured over the ceiling in the grill:
        761 px against 759.5)."""
        from packages.epicycle_annotator.add_preview import (
            ADD_MARKER_LIST_ROWS,
        )

        self._state_7f(widget)
        try:
            lst = widget.add_click_list
            assert not lst.isHidden()
            assert lst.count() == 12
            assert not widget.add_spacing_note.isHidden()
            assert not widget._add_membrane_only_label.isHidden()
            assert widget._add_result_container.isHidden()
            row = max(16, lst.sizeHintForRow(0))
            assert lst.height() // row == ADD_MARKER_LIST_ROWS
            viewport = self._screen_h(widget)
            busy, idle = self._busy_and_idle(widget)
            content = max(busy, idle)
            print("[ADD-HEIGHT] 7f: busy {} / idle {} px in {} px = {:.3f} "
                  "screens".format(busy, idle, viewport,
                                   content / max(viewport, 1)))
            assert content <= self.ADD_CANDIDATES_SCREENS * viewport, (
                "Add 7f is {} px in a {} px screen -- never raise 3.1".format(
                    content, viewport))
            assert content == self.MEASURED["7f"], content
            width = _slots(widget, "Add")["content"].minimumSizeHint().width()
            assert width <= self._viewport_w(widget), width
        finally:
            self._undo_7f(widget)

    def test_four_marker_rows_are_recorded_not_used(self, widget):
        """The measurement behind ADD_MARKER_LIST_ROWS = 3: printed, and
        pinned to be taller than 3 rows.  Measured IDLE (the taller state,
        C-MED): 4 rows are 761 px, over 759.5 -- the cap stays 3."""
        self._state_7f(widget)
        try:
            three = max(self._busy_and_idle(widget))
        finally:
            self._undo_7f(widget)
        self._state_7f(widget, rows=4)
        try:
            four = max(self._busy_and_idle(widget))
            viewport = self._screen_h(widget)
            print("[ADD-HEIGHT] 7f with 4 rows: {} px (3 rows {} px) in "
                  "{} px".format(four, three, viewport))
            assert four > three
        finally:
            self._undo_7f(widget)

    def test_the_dark_off_note_state_is_within_the_ceiling(self, widget):
        """D-1, state 7f+dark: 7f with "Treat dark space as outside" OFF.
        The two amber notes stacked measured 812 px (over), so the dark-off
        note wins S1's slot and the spacing note moves into the Advanced
        help slot, under ADD_DARK_OFF."""
        from packages.epicycle_annotator import add_preview as AP

        from packages.epicycle_annotator import (
            add_preview as _AP,
        )

        self._state_7f(widget)
        # No outline picked (the default outline decides what is outside
        # and greys the dark tick): the dark tick is the user's again
        widget.add_outline_combo.setCurrentIndex(
            _AP.OUTLINE_KEYS.index(_AP.OUTLINE_NONE))
        widget.add_dark_outside_cb.setChecked(False)
        try:
            assert not widget.add_dark_off_note.isHidden()
            assert widget.add_dark_off_note.text() == (
                AP.ADD_NOTE_DARK_OFF_LEAK)
            assert widget.add_spacing_note.isHidden(), "moved, not stacked"
            adv = widget._add_advanced_help.text()
            assert AP.ADD_DARK_OFF in adv
            assert AP.ADD_NOTE_DEFAULT_SPACING_BOXED in adv
            viewport = self._screen_h(widget)
            busy, idle = self._busy_and_idle(widget)
            content = max(busy, idle)
            print("[ADD-HEIGHT] 7f+dark: busy {} / idle {} px in {} px = "
                  "{:.3f} screens".format(busy, idle, viewport,
                                          content / max(viewport, 1)))
            assert content <= self.ADD_CANDIDATES_SCREENS * viewport, (
                "Add 7f+dark is {} px in a {} px screen -- never raise "
                "3.1".format(content, viewport))
            assert content == self.MEASURED["7f+dark"], content
            width = _slots(widget, "Add")["content"].minimumSizeHint().width()
            assert width <= self._viewport_w(widget), width
            # dark back on: the spacing note returns to S1
            widget.add_dark_outside_cb.setChecked(True)
            assert widget.add_dark_off_note.isHidden()
            assert not widget.add_spacing_note.isHidden()
        finally:
            self._undo_7f(widget)

    @pytest.mark.parametrize("attr", ["add_click_list", "add_candidate_list"])
    def test_the_add_lists_take_no_focus_and_no_selection(self, widget, attr):
        """A tick never takes the keyboard off the side
        view, so Shift+arrow keeps nudging the box -- and neither does the
        page's scroll area (a press on a NoFocus list hands the focus to
        the first ancestor that takes click focus)."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QAbstractItemView

        lst = getattr(widget, attr)
        assert lst.focusPolicy() == Qt.NoFocus
        assert lst.selectionMode() == QAbstractItemView.NoSelection
        assert _slots(widget, "Add")["scroll"].focusPolicy() == Qt.NoFocus

    def test_the_strip_is_pinned_and_does_not_move(self, widget):
        app = self._state(widget, "7a")
        try:
            h0 = widget._add_s4_container.height()
            widget.update_add_ui_state(busy=True, stage_text="Finding markers...")
            app.processEvents()
            assert widget._add_s4_container.height() == h0
            widget.update_add_ui_state(busy=False)
            app.processEvents()
            assert widget._add_s4_container.height() == h0
        finally:
            widget.hide()

    def test_clear_objects_costs_no_height_and_fits_the_row(self, widget):
        """Clear objects sits in the pinned Preview row: the strip's height
        is fixed, the scroll content does not move, and the row (Clear
        objects, Preview, Stop) fits the 312 px live width.  Measured on
        the tallest state that can show it (7f+dark: a box, no
        candidates), idle -- the state the button is shown in."""
        from qtpy.QtWidgets import QApplication

        self._state_7f(widget)
        widget.add_dark_outside_cb.setChecked(False)
        try:
            widget.update_add_ui_state(busy=False)
            for _ in range(3):
                QApplication.processEvents()
            assert not widget.add_clear_btn.isHidden()
            h_strip = widget._add_s4_container.height()
            content = self._content_h(widget)
            row = widget.add_preview_btn.parentWidget()
            row_w = row.minimumSizeHint().width()
            row_h = row.sizeHint().height()
            widget.add_clear_btn.setVisible(False)
            for _ in range(3):
                QApplication.processEvents()
            assert widget._add_s4_container.height() == h_strip
            assert self._content_h(widget) == content
            assert row.sizeHint().height() == row_h
            viewport = self._screen_h(widget)
            print("[ADD-HEIGHT] 7f+dark with Clear objects: {} px in {} px; "
                  "the Preview row needs {} px".format(content, viewport,
                                                       row_w))
            assert content <= self.ADD_CANDIDATES_SCREENS * viewport
            assert row_w <= self.LIVE_VIEWPORT_W, row_w
        finally:
            widget._apply_add_visibility()
            self._undo_7f(widget)

    # ------------------------------------------------------------------ #
    # Nuclei inside each cell of another run, and the whole patch
    # ------------------------------------------------------------------ #
    #: Measured 2026-11-12 (offscreen, 312 px, 245 px screen), busy / idle,
    #: after the page fix pass (the two reviews):
    #:   7d      728 / 728 px  2.97  Nuclei ▸ Anywhere, candidates (the
    #:           Look for row added +32, the threshold help off over
    #:           candidates: 716; the Nuclei help naming Inside each cell,
    #:           three lines: +12)
    #:   7g      574 / 574 px  2.34  Nuclei ▸ Inside, candidates, every
    #:           chip and note (562 before: the Nuclei help, +12)
    #:   7h      580 / 630 px  2.57  Nuclei ▸ Inside, box, before a
    #:           result, the count pending, the amber slot and the help
    #:           line at their longest (the help: the run marked nuclei,
    #:           the no-run how-to or a refused channel's reason; the
    #:           amber: half size) -- 618 before
    #:   7h0     575 / 625 px  2.55  Nuclei ▸ Anywhere, box, before a
    #:           result, no other run: the how-to is the grey item's
    #:           tooltip now, not a page line (700 before: the line and
    #:           a whole-patch amber Anywhere can no longer show)
    #:   REVIEW  401 / 401 px  1.64  12 added objects, every one with
    #:           both chips, S1 at its longest (the two-nuclei tail and
    #:           the select sentence) -- first measured here; 371 / 371
    #:           (1.51) since the page redesign (2026-11-14): the card's
    #:           Refine all ▸ · Refine ▸ · + Add another share one row and
    #:           Undo all moved to the footer.  Every other state here
    #:           re-measured unchanged after the redesign (R1-R6, the words)
    #: Cells (2026-11-12, the tissue outline and the whole patch), busy /
    #: idle:
    #:   7i      638 / 638 px  2.60  Cells ▸ Whole patch, Markers ▸ a run
    #:           (its count "in the patch"), candidates, every note
    #:   7j      731 / 731 px  2.98  Cells, Walls + prepare open (every
    #:           tick), Tissue outline + Show ticked (its legend),
    #:           candidates (the outline help hidden over them)
    #:   7k      558 / 596 px  2.43  Cells ▸ Whole patch, Markers ▸ a run,
    #:           before a result, the amber slot and S1 at their longest
    #:   7l      700 / 750 px  3.06  Cells, a box before a result, Walls +
    #:           prepare open, Tissue outline + its help + Show ticked (its
    #:           legend), the amber slot and S1 at their longest -- the
    #:           tallest derive state, 9.5 px under the ceiling
    #:   7l+sp   700 / 750 px  3.06  7l at the default spacing: the
    #:           derive note keeps the one amber slot, the spacing note is
    #:           in the Advanced help slot (two ambers stacked were 799)
    #:   7l+dark 648 / 698 px  2.85  7l with No outline and the dark tick
    #:           off: the dark-off note gives way (its reason is in the
    #:           Advanced help slot); Show and its legend go with the
    #:           outline (two ambers stacked were 759)
    #:   7l+both 648 / 698 px  2.85  both conditions at once
    #: (2026-11-24, the Cells whole patch runs at the never-set spacing:)
    #:   7k+sp   558 / 596 px  2.43  7k at the default spacing: the derive
    #:           note keeps the slot, the spacing note is in the Advanced
    #:           help slot (= 7k with Advanced closed)
    #:   7k+sp0  509 / 547 px  2.23  7k at the default spacing with no
    #:           derive note: the spacing note under S1
    #:   (7k and 7h0 measured with the Inside / whole-patch spacing refusal
    #:   gone from their longest-S1 lists: 7k unchanged at 596, 7h0 647 ->
    #:   635 -- see NUC_OUTLINE_MEASURED)
    #: No take-off was needed.  The page's tallest state stays 7f+dark
    #: (751 px); 3.1 not raised.
    #: (2026-11-16: the nothing-ticked S1 line with the two-nuclei tail --
    #: it names All, 125 characters -- is one more S1 line in the states
    #: that can show it: 7d 728 -> 740, 7g 574 -> 586; the Cells states read
    #: the Cells line, which does not move them)
    CELLS_MEASURED = {"7i": 638, "7j": 731, "7k": 596, "7l": 750,
                      "7l+sp": 750, "7l+dark": 698, "7l+both": 698,
                      "7k+sp": 596, "7k+sp0": 547}
    #: 7h+sp: 7h at the never-set spacing with no derive note -- Inside
    #: each cell runs there, and the spacing amber shows under S1 (with a
    #: derive note up, 7h, it would move to the Advanced help slot)
    DERIVE_STATES = ("7d", "7g", "7h", "7h+sp", "7h0", "REVIEW", "7i", "7j", "7k",
                     "7k+sp", "7k+sp0", "7l", "7l+sp", "7l+dark", "7l+both",
                     "7m0", "7m1", "7m1+sp", "7m1+auto", "7mc", "7mw",
                     "7n0", "7n1", "7n1+sp", "7n1+auto", "7n1cl", "7n1no",
                     "7n2", "7n2+sp", "7n2+auto", "7n3", "7nd", "7nmc",
                     "7nw0", "7nw1+auto")
    #: Nuclei ▸ the Tissue outline (an option, off by default), busy /
    #: idle (the larger counts):
    #:   7h0      Anywhere before a result (above): the outline card's
    #:            chevron, closed, under the Brightness card
    #:   7n0      Anywhere, the shrink-wrap chosen, its card CLOSED, the
    #:            histogram drawn, the default-spacing amber, the longest
    #:            Auto status and S1 at its longest (the outline's refusals)
    #:   7n1      Anywhere, the card OPEN (the histogram folds), the
    #:            shrink-wrap, Show ticked and its longest legend
    #:   7n1+sp   7n1 + the default-spacing amber
    #:   7n1+auto 7n1+sp + the longest Auto status
    #:   7n1cl    7n1 with the outline of the objects you cleared
    #:   7n1no    the card open with No outline (no Show)
    #:   7n2      the markers, 12 markers, the outline card open (Markers
    #:            closed), the shrink-wrap, Show + legend; +sp; +auto
    #:   7n3      the markers, 12 markers, Markers OPEN (its list), the
    #:            shrink-wrap chosen (its card closed), the spacing amber,
    #:            an Auto status, the outline's S1
    #:   7nd      7d with the shrink-wrap chosen (no Nuclei card over
    #:            candidates: 7d's rows)
    #:   7nmc     7mc with the shrink-wrap chosen (the Markers card only)
    #: Measured 2026-11-16 (offscreen, 312 px, 245 px screen), idle (the
    #: larger): 7h0 625 -> 647 (the closed chevron and the container's
    #: spacing) · 7n0 696 · 7n1 579 · 7n1+sp 640 · 7n1+auto 665 · 7n1cl 579
    #: · 7n1no 527 · 7n2 600 · 7n2+sp 661 · 7n2+auto 686 · 7n3 727 · 7nd 740
    #: (= 7d) · 7nmc 690 (= 7mc).  With the card's help line ON the page,
    #: 7n2+auto measured 759 and 7n1+auto 738 (the help is 73 px): the
    #: take-off moved what each choice does into its item's tooltip.  The
    #: tallest, 7n3, is 26 px under 7f+dark (753).
    #: (2026-11-22: the clicks' hint line stays on the markers at every
    #: marker count -- 47 px here, two lines with real fonts -- and the
    #: preset's help line goes while markers are placed: 7n3 727 -> 776
    #: with the hint alone, over the 3.1 ceiling, -> 738 with the preset
    #: help off; 7n2 600 -> 562, 7n2+sp 661 -> 623, 7n2+auto 686 -> 648,
    #: 7nmc 690 -> 652 (the preset help off, the hint not over candidates).
    #: The derive states' tallest stays 7d / 7nd, 740.)
    #: (2026-11-22, later: the hint line stays over a result too -- live,
    #: a result is on screen from the first click on: 7nmc 652 -> 701.)
    #: (2026-11-24: Inside each cell no longer refuses at the never-set
    #: spacing, so its 138-character refusal left the longest S1 an Inside
    #: state can show: 7h0 647 -> 635 (7h, unpinned, measured 568 / 618).)
    #: (2026-11-24, one region rule: Anywhere runs on a whole patch with its
    #: Tissue outline card: 7nw0 = 7n0 on a whole patch with the slow-patch
    #: amber at its longest; 7nw1+auto = 7n1+auto there.)
    NUC_OUTLINE_MEASURED = {"7h0": 635, "7n0": 696, "7n1": 579,
                            "7n1+sp": 640, "7n1+auto": 665, "7n1cl": 579,
                            "7n1no": 527, "7n2": 562, "7n2+sp": 623,
                            "7n2+auto": 648, "7n3": 738, "7nd": 740,
                            "7nmc": 701, "7nw0": 685, "7nw1+auto": 654}
    NUC_OUTLINE_STATES = ("7n0", "7n1", "7n1+sp", "7n1+auto", "7n1cl",
                          "7n1no", "7n2", "7n2+sp", "7n2+auto", "7n3", "7nd",
                          "7nmc", "7nw0", "7nw1+auto")
    #: Nuclei ▸ At the markers I place, busy / idle (the larger counts):
    #:   7m0      the markers picked, Markers open, no marker yet (the
    #:            Nuclei click help shown), S1 at its longest
    #:   7m1      12 markers (the widest rows, the list at 3 rows), the
    #:            clicks' hint line (it stays), the preset's help gone
    #:   7m1+sp   7m1 + the default-spacing amber (the Ball is in µm)
    #:   7m1+auto 7m1+sp + the longest Auto status under the threshold
    #:            (the threshold's own help folds with the histogram; an
    #:            Auto status does not)
    #:   7mc      the markers with candidates: Markers open with its
    #:            chevron and click rows (the list is hidden), the marks of
    #:            a dim and a kept marker, the default spacing and a flat
    #:            Auto status pushed (neither shows over candidates), the
    #:            counts note with every part, S1 at the longest line that
    #:            can stand above a result (a marker click refused over
    #:            candidates keeps them)
    #:   7mw      the markers on a whole patch (they are placed there as in
    #:            a box): 7m1+auto's rows -- 12 markers, the default
    #:            spacing, the longest Auto status -- with the slow-patch
    #:            amber at its longest (it takes the one amber slot; the
    #:            spacing note moves into Advanced) and the longest S1 a
    #:            whole patch with markers can show
    #: Measured 2026-11-16 (offscreen, 312 px, 245 px screen), busy / idle:
    #: 7m0 537 / 587 · 7m1 558 / 608 · 7m1+sp 619 / 669 · 7m1+auto 644 /
    #: 694 · 7mc 690 / 690 · 7mw 476 / 514.  The tallest, 7m1+auto, is
    #: 59 px under 7f+dark (753) and 65.5 px under the 3.1 ceiling; 7d and
    #: every Cells number above did not move.  (2026-11-16, the fix pass:
    #: 7mc built at its real worst case measures what it did, 690 -- the
    #: real orchestrator driven into a real page after a markers Preview
    #: shows the same rows; 7mw gained the slow-patch amber a big whole
    #: patch shows, 452 -> 514.  Live, at 312 px, the page measured 8-13
    #: px over these numbers and 7mc 732 against 690: offscreen Qt here
    #: has no fonts and no napari theme (row heights, button padding), and
    #: 7mc holds the most such rows -- its candidate list's four rows alone
    #: are 90 px offscreen and 106 px live.)
    #: 7m0 .. 7m1+auto grew 21 px when the tissue-outline card's closed
    #: chevron joined the Nuclei page (they were 587 / 608 / 669 / 694).
    #: (2026-11-21: markers are placed on a whole patch too, so 7mw now
    #: shows the Markers card with 12 markers, the longest Auto status and
    #: the slow-patch amber (the spacing amber moves into Advanced): 645 /
    #: 683, was 476 / 514 with the card hidden.  Still 32 px under 7m1+auto
    #: -- the whole patch has no Tissue outline chevron -- and 76.5 px under
    #: the 3.1 ceiling; no other state moved.)
    #: (2026-11-22: the clicks' hint line stays at every marker count, 47
    #: px here; with it alone 7m1+auto measured 764, over the 3.1 ceiling,
    #: so the preset's help line goes while markers are placed: 7m1 629 ->
    #: 640, 7m1+sp 690 -> 701, 7m1+auto 715 -> 726, 7mw 683 -> 694, 7mc
    #: 690 -> 652 (no hint over candidates, the preset help off); 7m0 has
    #: no marker and did not move.)
    #: (2026-11-22, later: the hint line stays over a result too -- live,
    #: a result is on screen from the first click on: 7mc 652 -> 701.)
    #: (2026-11-24, one region rule: the Tissue outline card is offered on
    #: a whole patch too, so 7mw shows its closed chevron, and Anywhere's
    #: whole-patch refusal left its longest S1: 7mw 694 -> 715.)
    MARKERS_MEASURED = {"7m0": 608, "7m1": 640, "7m1+sp": 701,
                        "7m1+auto": 726, "7mc": 701, "7mw": 715}
    MARKERS_STATES = ("7m0", "7m1", "7m1+sp", "7m1+auto", "7mc", "7mw")
    #: Cells, measured 2026-11-12 the same way, busy / idle -- see
    #: CELLS_MEASURED below; the page's tallest state stays 7f+dark.  The
    #: 7l variants add the two other amber notes' conditions (the default
    #: spacing; the dark tick off with no outline; both): the derive note
    #: keeps the ONE amber slot and the others move to the Advanced help
    #: slot.
    #: 7k+sp / 7k+sp0: a Cells whole patch at the default spacing runs as a
    #: box does and shows the spacing note -- in the Advanced help slot
    #: while the derive note holds the one amber slot (7k+sp), under S1
    #: when no derive note is up (7k+sp0).
    CELLS_STATES = ("7i", "7j", "7k", "7k+sp", "7k+sp0", "7l", "7l+sp",
                    "7l+dark", "7l+both")

    @staticmethod
    def _longest_cells_s1():
        """The longest S1 a Cells box or whole patch can show before a
        result: the whole patch's ready line, every refusal and transient
        of the outline and the solver, the box limit (the spacing refuses
        nothing)."""
        from packages.epicycle_annotator import add_preview as AP

        longest_item = max((AP.ADD_OUTLINE_CLEARED, AP.ADD_OUTLINE_SHRINK,
                            AP.ADD_OUTLINE_NONE), key=len)
        return max((AP.ADD_S1_PATCH_PLAIN.format(z=999, y=9999, x=9999,
                                                 m="999.9", p=100),
                    AP.over_limit_line((999, 9999, 9999)),
                    AP.whole_too_big_line(999_999_999),
                    AP.ADD_WHOLE_NEEDS_NUMBA,
                    AP.ADD_WHOLE_NEEDS_NUMBA_CACHE,
                    AP.ADD_S1_WHOLE_PICK_RUN, AP.ADD_S1_WHOLE_NO_RUN,
                    # Click markers on a whole patch: its ready line
                    AP.box_line((999, 9999, 9999), preset="cells",
                                n_clicks=12, n_ticked=11, whole_patch=True),
                    AP.ADD_S1_RUN_NONE_HERE, AP.ADD_S1_RUN_NONE_HERE_PATCH,
                    AP.ADD_S1_RUN_NO_MARKERS,
                    AP.ADD_S1_RUN_NO_MARKERS_PATCH,
                    AP.ADD_S1_RUN_NO_MARKERS_EMPTY.format(
                        dark=AP.ADD_DARK_OUTSIDE_LABEL),
                    AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PATCH.format(
                        dark=AP.ADD_DARK_OUTSIDE_LABEL),
                    AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN,
                    AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN_PATCH,
                    AP.ADD_S1_NO_MARKERS_OUTLINE_CLEARED,
                    AP.ADD_S1_OUTLINE_FAILED,
                    AP.ADD_S1_STOPPED_PATCH, AP.ADD_S1_FAILED_PATCH,
                    AP.ADD_S1_SHOWING, AP.ADD_S1_MEASURING_PATCH,
                    AP.ADD_OUTLINE_PENDING, AP.ADD_S1_RUN_LOADING,
                    AP.ADD_S1_RUN_UNUSABLE, AP.ADD_S1_NO_MARKERS_OUTLINE,
                    AP.ADD_S1_OUTLINE_CLEARED_GONE.format(
                        choice=longest_item),
                    AP.ADD_S1_LEFT_WHOLE_CLAMPED,
                    AP.ADD_S1_WHOLE_MARKERS_CLEARED, AP.ADD_S1_WARMING,
                    AP.ADD_S1_STOPPED, AP.ADD_S1_FAILED,
                    AP.ADD_S1_BUSY_REFUSED, AP.ADD_S1_PARKED_REFUSED,
                    AP.ADD_S1_INPUTS_CHANGED,
                    # "Update as I click" over the live limit / not run as
                    # you click, on a box or a whole patch
                    AP.ADD_S1_LIVE_OVER_GATE_BOX.format(
                        n=AP.voxel_limit_words(
                            AP.ADD_LIVE_CELLS_MAX_VOXELS)),
                    AP.ADD_S1_LIVE_OVER_GATE_PATCH_SIZE.format(
                        n=AP.voxel_limit_words(
                            AP.ADD_LIVE_CELLS_MAX_VOXELS)),
                    AP.ADD_S1_LIVE_OVER_GATE_PATCH,
                    AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX,
                    AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_BOX,
                    AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_PATCH,
                    AP.ADD_S1_LIVE_PREPARING),
                   key=len)

    @staticmethod
    def _longest_cells_amber():
        from packages.epicycle_annotator import add_preview as AP
        from packages.epicycle_annotator.reshape import region as _R

        return max((AP.ADD_NOTE_HALFSIZE_BOXED,
                    AP.one_channel_note(spread=False, shrink=True),
                    AP.one_channel_note(spread=True, shrink=True),
                    AP.one_channel_note(spread=True, shrink=True, whole=True),
                    AP.ADD_AMBER_OUTLINE_DROPS.format(p=100),
                    AP.ADD_AMBER_OUTLINE_DROPS_CLEARED.format(p=100),
                    AP.ADD_NOTE_PATCH_SLOW_STEPS.format(
                        m="999.9", t=AP.wait_phrase(999)),
                    AP.ADD_NOTE_BOX_SLOW_STEPS.format(
                        m="999.9", t=AP.wait_phrase(999)),
                    # a whole patch filled as you click, at the live limit
                    AP.ADD_NOTE_PATCH_SLOW_LIVE_CELLS.format(
                        m=AP.millions(AP.ADD_LIVE_CELLS_MAX_VOXELS)),
                    # the longest ridge note a whole patch can show: at the
                    # fill's ceiling, with the longer ridge name
                    AP.ridges_whole_line(
                        "Thin-faint ridges",
                        int(_R.WHOLE_FILL_MAX_VOXELS), "meijering"),
                    AP.ADD_NOTE_RIDGES_BOX.format(kind="Thin-faint ridges"),
                    AP.ADD_NOTE_RIDGES_BOX_ONCE.format(
                        kind="Thin-faint ridges")),
                   key=len)

    def _cells_state(self, widget, state, one_channel=False):
        """7i Cells ▸ Whole patch, Markers ▸ a run, candidates; 7j Cells,
        Walls + prepare open (every tick), the outline row + Show ticked,
        candidates; 7k Cells ▸ Whole patch, Markers ▸ a run, before a
        result, the amber slot at its longest; 7l Cells, a box before a
        result, Walls + prepare open, the outline row + its help + Show
        ticked (its legend), the amber slot at its longest.
        ``one_channel``: the patch has one image channel (Nuclei masks
        grey, Bridge gaps at its one-channel default)."""
        from qtpy.QtWidgets import QApplication

        from packages.epicycle_annotator import add_preview as AP

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Add")
        widget.refresh_add_image_combo(["mem"] if one_channel
                                       else ["nuc", "mem"])
        widget.set_add_preset("cells")
        items = [
            (AP.ADD_ITEM_RUN.format(name="a-rather-long-run-name-here-xyz")
             + AP.ADD_ITEM_RAW_SUFFIX, True, ("labels", "r")),
            ((AP.ADD_ITEM_NUCLEI_GREY, False, ("nuclei_masks", None))
             if one_channel else
             (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None))),
            (AP.ADD_ITEM_CLICKS, True, ("paint", None)),
        ]
        whole = state.startswith(("7i", "7k"))
        widget.update_add_ui_state(marker_sources=items, box_active=True,
                                   voxel_size_known=True, whole_patch=whole,
                                   cleared_in_force=True,
                                   outline_pending=True,
                                   single_channel=bool(one_channel))
        if whole:
            widget.add_marker_source_combo.setCurrentIndex(0)
            widget.add_markers_btn.setChecked(True)
            widget.update_add_ui_state(run_count=(123456, 123456))
        else:
            widget.add_walls_btn.setChecked(True)
            widget.add_wall_prep_btn.setChecked(True)
            for key in ("zcorrect", "bgsub", "lift", "denoise"):
                getattr(widget, "add_prep_{}_cb".format(key)).setChecked(True)
            # the longest help line: the shrink-wrap's
            widget.add_outline_combo.setCurrentIndex(
                AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
            widget.add_show_outline_cb.setChecked(True)
            if state in ("7l+dark", "7l+both"):
                # no outline and the dark tick off: the dark-off note's
                # condition (Show is gone with the outline)
                widget.add_outline_combo.setCurrentIndex(
                    AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
                widget.add_dark_outside_cb.setChecked(False)
            if state in ("7l+sp", "7l+both"):
                # the spacing note's condition
                widget.update_add_ui_state(default_spacing=True)
        if state in ("7k+sp", "7k+sp0"):
            # a whole patch at the default spacing: it runs, with the note
            widget.update_add_ui_state(default_spacing=True)
        if state in ("7i", "7j"):
            ids = list(range(400, 412))
            widget.update_add_ui_state(candidate_small_ids={401, 402})
            widget.update_add_ui_state(
                candidate_ids=ids, candidate_sizes={i: 1234567 for i in ids},
                candidate_edge_ids=set() if whole else {400, 401})
            counts = {k: 12 for k, _m, _o in AP._COUNT_PARTS}
            widget.update_add_ui_state(
                notes=[("counts", AP.counts_note(counts), False)],
                walls_toggle_enabled=True, walls_toggle_reason="",
                derive_note="")
            widget.add_show_walls_cb.setChecked(True)
            s1 = self._longest_s1(False)
        else:
            widget.update_add_ui_state(
                candidate_ids=[], candidate_sizes={},
                candidate_edge_ids=set(),
                derive_note=("" if state == "7k+sp0"
                             else self._longest_cells_amber()))
            s1 = self._longest_cells_s1()
        widget.update_add_ui_state(status_text=s1, busy=False)
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        return QApplication

    def _undo_cells_state(self, widget):
        # the Nuclei Tissue outline card's twins (a module-scoped widget
        # elsewhere carries them across states)
        if hasattr(widget, "add_nuc_outline_combo"):
            widget.add_nuc_show_outline_cb.setChecked(False)
            widget.add_nuc_outline_combo.setCurrentIndex(
                widget.add_nuc_outline_combo.count() - 1)
            widget.add_nuc_outline_btn.setChecked(False)
        widget.add_dark_outside_cb.setChecked(True)
        widget.update_add_ui_state(default_spacing=False)
        widget.add_show_walls_cb.setChecked(False)
        widget.add_show_outline_cb.setChecked(False)
        for key in ("zcorrect", "bgsub", "lift", "denoise"):
            getattr(widget, "add_prep_{}_cb".format(key)).setChecked(
                key == "zcorrect")
        widget.add_wall_prep_btn.setChecked(False)
        widget.add_walls_btn.setChecked(False)
        widget.add_markers_btn.setChecked(False)
        widget.update_add_ui_state(
            busy=False, derive_note="", whole_patch=False,
            cleared_in_force=False, outline_pending=False,
            candidate_ids=[], candidate_sizes={}, candidate_edge_ids=set(),
            notes=[], run_count=(), single_channel=False)
        widget.refresh_add_image_combo(["nuc", "mem"])
        widget.add_outline_combo.setCurrentIndex(
            widget.add_outline_combo.count() - 1)
        widget.clear_add_touched()
        widget.set_add_preset("nuclei")
        widget.clear_add_touched()

    @staticmethod
    def _longest_inside_help():
        """The longest line the card can show under Look for nuclei on
        Inside: its help, the run-marked-nuclei help, the no-run how-to, or
        the reason the patch's levels refused the channel."""
        from packages.epicycle_annotator import add_derive as AD
        from packages.epicycle_annotator import add_preview as AP

        try:
            AD._check_otsu(0.0)
            levels = ""
        except ValueError as exc:
            levels = AP.plain_reason(str(exc))
        return max((AP.ADD_HELP_INSIDE, AP.ADD_HELP_INSIDE_PICKED_NUCLEI,
                    AP.ADD_HELP_INSIDE_NO_RUN, levels), key=len)

    def _derive_state(self, widget, state):
        from qtpy.QtWidgets import QApplication

        import numpy as np

        from packages.epicycle_annotator import add_preview as AP

        if state in self.CELLS_STATES:
            return self._cells_state(widget, state)
        if state in self.MARKERS_STATES:
            return self._markers_state(widget, state)
        if state in self.NUC_OUTLINE_STATES:
            return self._nuc_outline_state(widget, state)

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Add")
        widget.refresh_add_image_combo(["nuc", "mem"])
        widget.set_add_preset("nuclei")
        name = "a-rather-long-run-name-here-xyz"
        if state == "7h0":
            items = [(AP.ADD_LOOK_NO_RUN, False, (AP.LOOK_INSIDE, None),
                      False)]
        else:
            items = [(AP.look_inside_text(name, "cells", curated=False),
                      True, (AP.LOOK_INSIDE, "r"), False)]
        widget.update_add_ui_state(look_for_items=items, box_active=True,
                                   voxel_size_known=True)
        widget.add_look_for_combo.setCurrentIndex(
            1 if state in ("7g", "7h", "7h+sp", "REVIEW") else 0)
        if state in ("7d", "7h0"):
            widget.update_add_ui_state(histogram_data={
                "values": list(np.linspace(0.0, 1.0, 500)), "n_bins": 32})
        if state == "REVIEW":
            ids = list(range(400, 412))
            two_cut = [AP.ADD_CHIP_TWO, AP.ADD_CHIP_CUT]
            widget.update_add_ui_state(
                review_active=True, piece_ids=ids,
                piece_sizes={i: 1234567 for i in ids},
                piece_chips={i: list(two_cut) for i in ids},
                piece_tips={i: AP.ADD_TIP_TWO.format(cell=812, k=2)
                            for i in ids},
                add_batch_refine_enabled=True, start_enabled=True,
                candidate_ids=[], candidate_sizes={},
                candidate_edge_ids=set(), derive_note="")
            s1 = max((AP.added_line(12, 12345678, n_selected=n, n_two=12)
                      for n in (0, 1, 2)), key=len)
        elif state in ("7d", "7g"):
            ids = list(range(400, 412))
            flags = ({} if state == "7d" else
                     {400: ("two",), 401: ("two", "cut"), 402: ("cut",)})
            widget.update_add_ui_state(candidate_small_ids={403, 404},
                                       candidate_flags=flags,
                                       candidate_tips={})
            widget.update_add_ui_state(
                candidate_ids=ids, candidate_sizes={i: 1234567 for i in ids},
                candidate_edge_ids={404, 405})
            counts = {k: 12 for k, _m, _o in AP._COUNT_PARTS}
            widget.update_add_ui_state(
                notes=[("counts", AP.counts_note(counts), False)],
                walls_toggle_enabled=False,
                walls_toggle_reason=(AP.ADD_WALLS_INSIDE if state == "7g"
                                     else AP.WALLS_TOGGLE_NO_WALLS))
            s1 = max((self._longest_s1(False, two_nuclei=True),
                      AP.found_line(12, k_edge=2, k_small=2, k_both=1,
                                    n_two=12),
                      AP.ADD_S1_INSIDE_ALL_TAKEN,
                      AP.ADD_S1_INSIDE_ALL_TAKEN_NUCLEI_RUN,
                      AP.ADD_S1_INSIDE_NOTHING), key=len)
            widget.update_add_ui_state(derive_note="")
        else:
            widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                                       candidate_edge_ids=set())
            if state == "7h+sp":
                # the spacing amber alone in the slot (no derive note)
                widget.update_add_ui_state(inside_count=(), derive_note="",
                                           default_spacing=True)
                widget._add_source_note = self._longest_inside_help()
                widget._apply_add_visibility()
            elif state == "7h":
                widget.update_add_ui_state(
                    inside_count=(), derive_note=max(
                        (AP.ADD_NOTE_HALFSIZE_BOXED,
                         AP.ADD_AMBER_BOX_MOSTLY_FULL.format(p=19),
                         AP.ADD_AMBER_BOX_MOSTLY_FULL_NUCLEI_RUN.format(
                             p=19),
                         AP.ADD_NOTE_PATCH_SLOW_STEPS.format(
                             m="999.9", t=AP.wait_phrase(999)),
                         AP.ADD_NOTE_PATCH_SLOW_PLAIN.format(m="99999.9")),
                        key=len))
                # the longest help line the card can show on Inside
                widget._add_source_note = self._longest_inside_help()
                widget._apply_add_visibility()
            else:
                widget.update_add_ui_state(derive_note="")
            s1 = max((AP.ADD_S1_INSIDE_NO_CELLS,
                      AP.ADD_S1_INSIDE_NO_CELLS_PATCH,
                      AP.ADD_S1_INSIDE_UNUSABLE,
                      AP.ADD_S1_INSIDE_CHANNEL_UNUSABLE, AP.ADD_S1_RUN_LOADING,
                      AP.ADD_S1_PATCH_PLAIN.format(z=999, y=9999, x=9999,
                                                   m="999.9", p=100),
                      AP.ADD_S1_PICK_CHANNEL), key=len)
        widget.update_add_ui_state(status_text=s1, busy=False)
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        return QApplication

    @staticmethod
    def _longest_markers_s1():
        """The longest S1 Nuclei ▸ At the markers I place can show on a box
        before a result: its refusals, the click transients, the ready line
        with its ticked tail, and the shared transients."""
        from packages.epicycle_annotator import add_preview as AP

        return max((AP.ADD_S1_NO_MARKERS, AP.ADD_S1_NO_MARKERS_ON,
                    AP.nuclei_tick_one_line(12),
                    AP.ADD_S1_MARKERS_NONE_ON_STAIN,
                    AP.markers_all_kept_line(12),
                    AP.ADD_S1_CLICK_OUTSIDE,
                    AP.ADD_S1_CLICK_ON_KEPT.format(id=123456),
                    AP.ADD_S1_CLICK_SAME_SPOT.format(id=123456),
                    AP.ADD_S1_CLICK_SAME_SPOT_UNTICKED.format(id=123456),
                    AP.ADD_S1_CLICKS_DROPPED.format(n=12),
                    AP.ADD_S1_WHOLE_MARKERS_CLEARED,
                    AP.rebased_line(-123456, 12),
                    AP.box_line((999, 9999, 9999), preset="nuclei",
                                n_clicks=12, n_ticked=11,
                                look_for=AP.LOOK_MARKERS),
                    AP.ADD_S1_ALL_FILTERED, AP.ADD_S1_PICK_CHANNEL,
                    AP.ADD_S1_BUSY_REFUSED, AP.ADD_S1_PARKED_REFUSED,
                    AP.ADD_S1_FAILED, AP.ADD_S1_STOPPED,
                    AP.ADD_S1_INPUTS_CHANGED),
                   key=len)

    @staticmethod
    def _longest_nuclei_slow_line():
        """The longest slow-patch amber a Nuclei whole patch shows, the
        tick on or off."""
        from packages.epicycle_annotator import add_preview as AP

        return max((AP.nuclei_patch_slow_line(n, live=live, anywhere=any_)
                    for n in (AP.ADD_SLOW_PATCH_VOXELS + 1, 99_999_999,
                              999_999_999, 9_999_999_999)
                    for live in (False, True) for any_ in (False, True)),
                   key=len)

    def _markers_state(self, widget, state):
        """7m0 / 7m1 / 7m1+sp / 7m1+auto / 7mc / 7mw: Nuclei ▸ At the
        markers I place (see MARKERS_MEASURED)."""
        from qtpy.QtWidgets import QApplication

        import numpy as np

        from packages.epicycle_annotator import add_preview as AP

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Add")
        widget.refresh_add_image_combo(["nuc", "mem"])
        widget.set_add_preset("nuclei")
        name = "a-rather-long-run-name-here-xyz"
        items = [(AP.look_inside_text(name, "cells", curated=False),
                  True, (AP.LOOK_INSIDE, "r"), False)]
        whole = state == "7mw"
        widget.update_add_ui_state(look_for_items=items, box_active=True,
                                   voxel_size_known=True, whole_patch=whole)
        keys = list(widget._add_look_keys)
        widget.add_look_for_combo.setCurrentIndex(
            keys.index((AP.LOOK_MARKERS, None)))
        widget.update_add_ui_state(histogram_data={
            "values": list(np.linspace(0.0, 1.0, 500)), "n_bins": 32})
        if state in ("7m1", "7m1+sp", "7m1+auto", "7mc", "7mw"):
            markers = [(65520 + i, (1234, 1234, 1234), i % 3 != 1)
                       for i in range(12)]
            widget.update_add_ui_state(click_count=12, click_markers=markers)
        if state in ("7m1+sp", "7m1+auto", "7mw"):
            widget.update_add_ui_state(default_spacing=True)
        if state in ("7m1+auto", "7mw"):
            # the longest Auto status line (a flat box)
            widget.update_add_ui_state(brightness_flat=True)
        if state == "7mc":
            # the worst case the app can reach over candidates: 12 markers
            # (one dim, one on a kept object), Markers open, the default
            # spacing and a flat-box Auto status pushed (neither may show
            # over candidates), the counts note with every part, the
            # longest S1 that can stand above a result -- a marker click
            # refused over candidates keeps them up and says why
            widget.update_add_ui_state(
                click_absorbed_ids=[65520],
                click_absorbed_tips={65521: AP.ADD_MARKER_DIM_TIP},
                default_spacing=True, brightness_flat=True)
            ids = list(range(400, 412))
            widget.update_add_ui_state(candidate_small_ids={403, 404},
                                       candidate_flags={},
                                       candidate_tips={})
            widget.update_add_ui_state(
                candidate_ids=ids, candidate_sizes={i: 1234567 for i in ids},
                candidate_edge_ids={404, 405})
            counts = {k: 12 for k, _m, _o in AP._COUNT_PARTS}
            widget.update_add_ui_state(
                notes=[("counts", AP.counts_note(counts), False)],
                walls_toggle_enabled=False,
                walls_toggle_reason=AP.WALLS_TOGGLE_NO_WALLS,
                derive_note="")
            s1 = max((self._longest_s1(False),
                      AP.ADD_S1_CLICK_OUTSIDE,
                      AP.ADD_S1_CLICK_ON_KEPT.format(id=123456),
                      AP.ADD_S1_CLICK_SAME_SPOT.format(id=123456),
                      AP.ADD_S1_CLICK_SAME_SPOT_UNTICKED.format(id=123456)),
                     key=len)
        else:
            # a whole patch shows its slow-patch amber (the Nuclei derive
            # slot reads the patch's size, not Look for nuclei)
            widget.update_add_ui_state(
                candidate_ids=[], candidate_sizes={},
                candidate_edge_ids=set(),
                derive_note=(self._longest_nuclei_slow_line() if whole
                             else ""))
            if whole:
                s1 = max((self._longest_markers_s1(),
                          AP.box_line((999, 9999, 9999), preset="nuclei",
                                      n_clicks=12, n_ticked=11,
                                      look_for=AP.LOOK_MARKERS,
                                      whole_patch=True),
                          AP.ADD_S1_PATCH_PLAIN.format(
                              z=999, y=9999, x=9999, m="999.9", p=100),
                          AP.ADD_S1_PICK_CHANNEL, AP.ADD_S1_BUSY_REFUSED,
                          AP.ADD_S1_PARKED_REFUSED, AP.ADD_S1_STOPPED_PATCH,
                          AP.ADD_S1_FAILED_PATCH, AP.ADD_S1_INPUTS_CHANGED),
                         key=len)
            else:
                s1 = self._longest_markers_s1()
        widget.update_add_ui_state(status_text=s1, busy=False)
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        return QApplication

    def _undo_markers_state(self, widget):
        widget.update_add_ui_state(
            click_count=0, click_markers=[], click_absorbed_ids=[],
            click_absorbed_tips={},
            default_spacing=False, brightness_flat=False,
            candidate_ids=[], candidate_sizes={}, candidate_edge_ids=set(),
            candidate_small_ids=set(), notes=[])
        widget.add_markers_btn.setChecked(False)

    @staticmethod
    def _longest_nuc_outline_s1(markers, whole=False):
        """The longest S1 Nuclei ▸ Anywhere (or the markers) can show on a
        box before a result with a Tissue outline chosen: its own lines, the
        outline's refusals and waits, the shared transients."""
        from packages.epicycle_annotator import add_preview as AP

        if markers:
            own = (TestTheAddPageAfterRebuild._longest_markers_s1(),)
        else:
            own = (AP.ADD_S1_NOTHING_NUCLEI, AP.ADD_S1_ALL_FILTERED,
                   AP.ADD_S1_MEASURING, AP.ADD_S1_PICK_CHANNEL,
                   AP.box_line((999, 9999, 9999), preset="nuclei",
                               empty_share=1.0))
        if whole:
            # Anywhere on a whole patch: its ready line, its size refusal
            # and the patch's own transients
            own += (AP.ADD_S1_PATCH_PLAIN.format(z=999, y=9999, x=9999,
                                                 m="999.9", p=100),
                    AP.whole_too_big_line(9_999_999_999),
                    AP.ADD_S1_MEASURING_PATCH, AP.ADD_S1_STOPPED_PATCH,
                    AP.ADD_S1_FAILED_PATCH)
        return max(own + (AP.ADD_S1_NUCLEI_ALL_OUTSIDE,
                          AP.ADD_S1_NUCLEI_ALL_OUTSIDE_CLEARED,
                          AP.ADD_S1_OUTLINE_FAILED_NUCLEI,
                          AP.ADD_OUTLINE_PENDING, AP.ADD_OUTLINE_DRAWING,
                          AP.ADD_S1_OUTLINE_CLEARED_GONE.format(
                              choice=AP.ADD_OUTLINE_NONE),
                          AP.ADD_S1_BUSY_REFUSED, AP.ADD_S1_PARKED_REFUSED,
                          AP.ADD_S1_FAILED, AP.ADD_S1_STOPPED,
                          AP.ADD_S1_INPUTS_CHANGED),
                   key=len)

    def _nuc_outline_state(self, widget, state):
        """7n0 / 7n1* / 7n2* / 7nd / 7nmc: Nuclei with a Tissue outline
        (see NUC_OUTLINE_MEASURED)."""
        from qtpy.QtWidgets import QApplication

        import numpy as np

        from packages.epicycle_annotator import add_preview as AP

        shrink = AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK)
        if state in ("7nd", "7nmc"):
            if state == "7nd":
                self._derive_state(widget, "7d")
            else:
                self._markers_state(widget, "7mc")
            widget.add_nuc_outline_combo.setCurrentIndex(shrink)
            widget.update_add_ui_state(default_spacing=True)
            widget.resize(340, 350)
            for _ in range(3):
                QApplication.processEvents()
            return QApplication
        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Add")
        widget.refresh_add_image_combo(["nuc", "mem"])
        widget.set_add_preset("nuclei")
        name = "a-rather-long-run-name-here-xyz"
        items = [(AP.look_inside_text(name, "cells", curated=False),
                  True, (AP.LOOK_INSIDE, "r"), False)]
        markers = state.startswith(("7n2", "7n3"))
        # 7nw*: Anywhere on a whole patch -- the same card as a box, plus
        # the slow-patch amber at its longest
        whole = state.startswith("7nw")
        closed = state in ("7n0", "7n3", "7nw0")
        widget.update_add_ui_state(look_for_items=items, box_active=True,
                                   voxel_size_known=True,
                                   cleared_in_force=True, whole_patch=whole)
        keys = list(widget._add_look_keys)
        widget.add_look_for_combo.setCurrentIndex(keys.index(
            (AP.LOOK_MARKERS, None) if markers
            else (AP.LOOK_ANYWHERE, None)))
        widget.update_add_ui_state(histogram_data={
            "values": list(np.linspace(0.0, 1.0, 500)), "n_bins": 32})
        if markers:
            rows = [(65520 + i, (1234, 1234, 1234), i % 3 != 1)
                    for i in range(12)]
            widget.update_add_ui_state(click_count=12, click_markers=rows)
        choice = {"7n1cl": AP.OUTLINE_CLEARED,
                  "7n1no": AP.OUTLINE_NONE}.get(state, AP.OUTLINE_SHRINK)
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(choice))
        if not closed:
            widget.add_nuc_outline_btn.setChecked(True)
        if choice != AP.OUTLINE_NONE:
            widget.add_nuc_show_outline_cb.setChecked(True)
            # the longest legend: drawn in the main view only
            widget.update_add_ui_state(outline_drawn=True,
                                       outline_in_side_views=False)
        if closed or state.endswith(("+sp", "+auto")):
            widget.update_add_ui_state(default_spacing=True)
        if closed or state.endswith("+auto"):
            widget.update_add_ui_state(brightness_flat=True)
        widget.update_add_ui_state(
            candidate_ids=[], candidate_sizes={}, candidate_edge_ids=set(),
            derive_note=(self._longest_nuclei_slow_line() if whole else ""))
        widget.update_add_ui_state(
            status_text=self._longest_nuc_outline_s1(markers, whole),
            busy=False)
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        return QApplication

    def _undo_nuc_outline_state(self, widget):
        widget.add_nuc_show_outline_cb.setChecked(False)
        widget.add_nuc_outline_combo.setCurrentIndex(
            widget.add_nuc_outline_combo.count() - 1)
        widget.add_nuc_outline_btn.setChecked(False)
        widget.update_add_ui_state(
            click_count=0, click_markers=[], click_absorbed_ids=[],
            click_absorbed_tips={}, outline_drawn=True,
            outline_in_side_views=True, cleared_in_force=False,
            default_spacing=False, brightness_flat=False,
            candidate_ids=[], candidate_sizes={}, candidate_edge_ids=set(),
            candidate_small_ids=set(), notes=[])
        widget.add_markers_btn.setChecked(False)
        widget.clear_add_touched()

    def _derive_heights(self, widget):
        from qtpy.QtWidgets import QApplication

        out = []
        for busy in (True, False):
            widget.update_add_ui_state(busy=busy, stop_enabled=busy,
                                       stage_text="Finding nuclei...")
            for _ in range(3):
                QApplication.processEvents()
            out.append(self._content_h(widget))
        return tuple(out)

    def test_add_derive_states_fit(self, widget):
        """7d / 7g / 7h / 7h0 / REVIEW and the Cells states 7i / 7j / 7k /
        7l, idle and busy, each within 3.1 screens -- the tallest is named:
        7l, 750 px idle (Cells, a box before a result, Walls + prepare open,
        the outline row, its help and Show ticked, the amber slot and S1 at
        their longest); the page's tallest stays 7f+dark (751 px)."""
        heights = {}
        for state in self.DERIVE_STATES:
            self._derive_state(widget, state)
            try:
                if state == "7g":
                    assert widget._add_threshold_row.isHidden()
                    assert not widget._add_result_container.isHidden()
                    assert widget._add_inside_help.isHidden()
                if state == "7h":
                    assert not widget._add_inside_help.isHidden()
                    assert widget._add_inside_help.text() == (
                        self._longest_inside_help())
                    assert not widget._add_inside_count.isHidden()
                    assert not widget._add_derive_note.isHidden()
                if state == "7h+sp":
                    assert widget.add_look_inside()
                    assert widget._add_derive_note.isHidden()
                    assert not widget.add_spacing_note.isHidden()
                if state == "7h0":
                    # the how-to is the grey item's tooltip only
                    assert widget._add_inside_help.isHidden()
                    assert widget._add_derive_note.isHidden()
                if state == "7d":
                    assert widget._add_threshold_help.isHidden()
                    assert not widget._add_look_row.isHidden()
                if state == "REVIEW":
                    assert not widget._add_review_container.isHidden()
                    assert widget._add_tune_container.isHidden()
                    assert widget.add_pieces_list.count() == 12
                if state.startswith(("7i", "7k")):
                    # a whole patch: Grow by shown (the user's pick there
                    # too), the run's count shown
                    assert not widget._add_grow_row.isHidden()
                    assert not widget.add_run_count_label.isHidden()
                    from packages.epicycle_annotator.add_preview import (
                        ADD_HELP_WHOLE,
                    )

                    assert widget._add_box_help.text() == ADD_HELP_WHOLE
                if state in ("7j", "7l", "7l+sp"):
                    assert not widget._add_outline_row.isHidden()
                    assert not widget.add_show_outline_cb.isHidden()
                    assert not widget._add_show_outline_legend.isHidden()
                    assert not widget._add_wall_prep_container.isHidden()
                if state == "7j":
                    assert widget._add_outline_help.isHidden()
                if state.startswith("7l"):
                    assert not widget._add_outline_help.isHidden()
                if state in ("7k", "7k+sp") or state.startswith("7l"):
                    assert not widget._add_derive_note.isHidden()
                    assert widget._add_derive_note.text() == (
                        self._longest_cells_amber())
                    assert widget._add_result_container.isHidden()
                if state in self.CELLS_STATES:
                    # ONE amber slot: two notes that each fit do not fit
                    # together
                    ambers = [w for w in (widget.add_spacing_note,
                                          widget.add_dark_off_note,
                                          widget._add_derive_note)
                              if not w.isHidden()]
                    assert len(ambers) == (
                        1 if state.startswith(("7k", "7l"))
                        else 0), (state, [w.text() for w in ambers])
                if state == "7k+sp0":
                    # a whole patch at the default spacing, no derive note:
                    # the spacing note under S1, as on a box
                    assert widget._add_derive_note.isHidden()
                    assert not widget.add_spacing_note.isHidden()
                if state == "7k+sp":
                    # the derive note keeps the slot: the spacing note is
                    # in the Advanced help slot
                    from packages.epicycle_annotator import (
                        add_preview as _AP,
                    )

                    assert widget.add_spacing_note.isHidden()
                    assert _AP.ADD_NOTE_DEFAULT_SPACING_BOXED in (
                        widget._add_advanced_help.text())
                if state in ("7l+sp", "7l+both"):
                    # the spacing note moved to the Advanced help slot
                    from packages.epicycle_annotator import (
                        add_preview as _AP,
                    )

                    assert _AP.ADD_NOTE_DEFAULT_SPACING_BOXED in (
                        widget._add_advanced_help.text())
                if state in ("7i", "7j"):
                    assert not widget._add_result_container.isHidden()
                    assert widget.add_candidate_list.count() == 12
                if state in self.MARKERS_STATES:
                    plot = (getattr(widget, "_add_plot_widget", None)
                            or widget._add_plot_placeholder)
                    # the histogram folds while the markers are the method
                    assert plot.isHidden()
                    assert widget._add_walls_container.isHidden()
                    assert not widget._add_threshold_row.isHidden()
                    if state == "7mw":
                        # the whole patch shows the Markers card as a box
                        # does, and its slow-patch amber
                        assert not widget._add_cells_container.isHidden()
                        assert not widget._add_derive_note.isHidden()
                        assert widget._add_derive_note.text() == (
                            self._longest_nuclei_slow_line())
                    assert not widget._add_markers_container.isHidden()
                    assert not widget._add_click_rows.isHidden()
                    assert widget._add_source_row.isHidden()
                    # the clicks' hint line: at every marker count, before
                    # and over a result
                    assert not widget._add_source_help.isHidden()
                    assert (widget.add_click_list.isHidden()
                            is (state in ("7m0", "7mc")))
                    assert (widget._add_result_container.isHidden()
                            is (state != "7mc"))
                    assert (widget.add_spacing_note.isHidden()
                            is (state not in ("7m1+sp", "7m1+auto")))
                    assert (widget._add_threshold_help.isHidden()
                            is (state not in ("7m1+auto", "7mw")))
                if state in self.MARKERS_STATES + ("7h0",) and (
                        hasattr(widget, "add_nuc_outline_btn")):
                    # the Tissue outline card: its chevron, closed, before
                    # a result on a box or a whole patch; never over
                    # candidates
                    assert (widget.add_nuc_outline_btn.isHidden()
                            is (state == "7mc"))
                    assert widget._add_nuc_outline_container.isHidden()
                if state in self.NUC_OUTLINE_STATES:
                    out_btn = widget.add_nuc_outline_btn
                    body = widget._add_nuc_outline_container
                    plot = (getattr(widget, "_add_plot_widget", None)
                            or widget._add_plot_placeholder)
                    if state in ("7nd", "7nmc"):
                        assert out_btn.isHidden() and body.isHidden()
                        assert not widget._add_result_container.isHidden()
                        assert widget.add_spacing_note.isHidden()
                    else:
                        closed = state in ("7n0", "7n3", "7nw0")
                        assert not out_btn.isHidden()
                        assert body.isHidden() is closed
                        # an open Nuclei card folds the histogram (the
                        # markers fold it anyway)
                        assert plot.isHidden() is (
                            state not in ("7n0", "7nw0"))
                        if state.startswith("7nw"):
                            # the whole patch: the same card as a box, and
                            # its slow-patch amber
                            assert not widget._add_derive_note.isHidden()
                        # the help line is off the page (the take-off)
                        assert widget._add_nuc_outline_help.isHidden()
                        legend = widget._add_nuc_show_outline_legend
                        # (on a whole patch the slow-patch amber holds
                        # the one amber slot: the spacing note moves into
                        # the Advanced help slot)
                        assert (widget.add_spacing_note.isHidden()
                                is not ((closed or state.endswith(
                                    ("+sp", "+auto")))
                                        and not state.startswith("7nw")))
                        if closed:
                            # Show and its legend live in the card's body:
                            # with the card closed neither is on screen
                            # (their own flags follow the tick, as Cells')
                            assert not legend.isVisible()
                            assert not (widget.add_nuc_show_outline_cb
                                        .isVisible())
                        else:
                            assert legend.isHidden() is (state == "7n1no")
                            assert (widget.add_nuc_show_outline_cb.isHidden()
                                    is (state == "7n1no"))
                    if state.startswith("7n2"):
                        assert widget._add_markers_container.isHidden()
                        assert not widget.add_markers_btn.isHidden()
                    if state == "7n3":
                        assert not widget._add_markers_container.isHidden()
                        assert not widget.add_click_list.isHidden()
                busy, idle = self._derive_heights(widget)
                viewport = self._screen_h(widget)
                heights[state] = max(busy, idle)
                print("[ADD-HEIGHT] {}: busy {} / idle {} px in {} px = "
                      "{:.3f} screens".format(state, busy, idle, viewport,
                                              heights[state] / viewport))
                width = _slots(widget, "Add")["content"].minimumSizeHint(
                ).width()
                assert width <= self._viewport_w(widget), (state, width)
                assert heights[state] <= (
                    self.ADD_CANDIDATES_SCREENS * viewport), (
                    "Add {} is {} px in a {} px screen -- never raise "
                    "3.1".format(state, heights[state], viewport))
                if state == "7d":
                    # 7d is also one of the candidates states: one number
                    assert heights[state] == self.MEASURED["7d"]
                if state in self.CELLS_MEASURED:
                    assert heights[state] == self.CELLS_MEASURED[state], (
                        "Add {} moved: {} px, recorded {} -- record the new "
                        "number with the reason".format(
                            state, heights[state],
                            self.CELLS_MEASURED[state]))
                if state in self.MARKERS_MEASURED:
                    assert heights[state] == self.MARKERS_MEASURED[state], (
                        "Add {} moved: {} px, recorded {} -- record the new "
                        "number with the reason".format(
                            state, heights[state],
                            self.MARKERS_MEASURED[state]))
                if state in self.NUC_OUTLINE_MEASURED:
                    assert heights[state] == (
                        self.NUC_OUTLINE_MEASURED[state]), (
                        "Add {} moved: {} px, recorded {} -- record the new "
                        "number with the reason".format(
                            state, heights[state],
                            self.NUC_OUTLINE_MEASURED[state]))
            finally:
                if state in self.CELLS_STATES:
                    self._undo_cells_state(widget)
                if state in self.MARKERS_STATES:
                    self._undo_markers_state(widget)
                if state in self.NUC_OUTLINE_STATES:
                    self._undo_nuc_outline_state(widget)
                widget._add_source_note = ""
                widget.add_look_for_combo.setCurrentIndex(0)
                widget.update_add_ui_state(busy=False, derive_note="",
                                           whole_patch=False,
                                           default_spacing=False,
                                           review_active=False, piece_ids=[],
                                           piece_sizes={}, piece_chips={},
                                           piece_tips={})
                widget.hide()
        tallest = max(heights, key=heights.get)
        print("[ADD-HEIGHT] tallest of the derive states: {} ({} px)".format(
            tallest, heights[tallest]))
        assert tallest in self.DERIVE_STATES
        assert heights[tallest] < self.MEASURED["7f+dark"], (
            "the page's tallest state is 7f+dark")

    def test_add_nuclei_outline_states_fit(self, widget):
        """The Nuclei Tissue outline states on their own (they are derive
        states too): each pinned, each under 7f+dark (753 px) and the 3.1
        ceiling."""
        for state in ("7h0",) + self.NUC_OUTLINE_STATES:
            self._derive_state(widget, state)
            try:
                busy, idle = self._derive_heights(widget)
                viewport = self._screen_h(widget)
                h = max(busy, idle)
                print("[ADD-HEIGHT] {}: busy {} / idle {} px".format(
                    state, busy, idle))
                width = _slots(widget, "Add")["content"].minimumSizeHint(
                ).width()
                assert width <= self._viewport_w(widget), (state, width)
                assert h <= self.ADD_CANDIDATES_SCREENS * viewport, state
                assert h < self.MEASURED["7f+dark"], state
                assert h == self.NUC_OUTLINE_MEASURED[state], (
                    "Add {} moved: {} px, recorded {} -- record the new "
                    "number with the reason".format(
                        state, h, self.NUC_OUTLINE_MEASURED[state]))
            finally:
                if state in self.NUC_OUTLINE_STATES:
                    self._undo_nuc_outline_state(widget)
                widget._add_source_note = ""
                widget.add_look_for_combo.setCurrentIndex(0)
                widget.update_add_ui_state(busy=False, derive_note="",
                                           whole_patch=False)
                widget.hide()

    def _nuclei_bodies_open(self, widget):
        return [b for b in ("_add_markers_container",
                            "_add_nuc_outline_container")
                if not getattr(widget, b).isHidden()]

    @pytest.mark.parametrize("sequence", ("candidates", "whole"))
    def test_two_nuclei_cards_never_open_together(self, widget, sequence):
        """A card hidden for a moment (candidates up, a whole patch) is
        still one of this preset's cards: opening another closes it, so it
        never comes back open beside the one opened last -- and the page
        is never two bodies tall.

        "candidates": the markers, Tissue outline opened (Markers closes),
        a Preview (the outline's chevron hides), Markers opened, Discard.
        "whole": Anywhere, Tissue outline opened, Whole patch, the markers
        picked (Markers opens), S / M / L (a box again)."""
        from qtpy.QtWidgets import QApplication

        from packages.epicycle_annotator import add_preview as AP

        shrink = AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK)
        if sequence == "candidates":
            self._markers_state(widget, "7m1")
            widget.add_nuc_outline_combo.setCurrentIndex(shrink)
            widget.add_nuc_outline_btn.setChecked(True)
            assert not widget.add_markers_btn.isChecked()
            ids = list(range(400, 412))
            widget.update_add_ui_state(
                candidate_ids=ids, candidate_sizes={i: 1234 for i in ids},
                candidate_edge_ids=set())
            assert widget.add_nuc_outline_btn.isHidden()
            widget.add_markers_btn.setChecked(True)
            widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                                       candidate_edge_ids=set())
        else:
            self._nuc_outline_state(widget, "7n1")
            assert widget.add_nuc_outline_btn.isChecked()
            widget.update_add_ui_state(whole_patch=True)
            # the card stays on a whole patch, as on a box
            assert not widget.add_nuc_outline_btn.isHidden()
            keys = list(widget._add_look_keys)
            widget.add_look_for_combo.setCurrentIndex(
                keys.index((AP.LOOK_MARKERS, None)))
            assert widget.add_markers_btn.isChecked()
            widget.update_add_ui_state(whole_patch=False)
        for _ in range(3):
            QApplication.processEvents()
        try:
            shown = not widget.add_nuc_outline_btn.isHidden()
            bodies = self._nuclei_bodies_open(widget)
            still = widget.add_nuc_outline_btn.isChecked()
            h = max(self._derive_heights(widget))
            # closing the outline's card by hand changes nothing: it was
            # already closed
            widget.add_nuc_outline_btn.setChecked(False)
            h_one = max(self._derive_heights(widget))
            print("[ADD-HEIGHT] two-cards {}: {} px (one card {} px), "
                  "bodies {}".format(sequence, h, h_one, bodies))
            assert shown
            assert bodies == ["_add_markers_container"], (
                "the card opened last is the only one open")
            assert not still
            assert h == h_one
            assert h <= self.ADD_CANDIDATES_SCREENS * self._screen_h(widget)
            assert h < self.MEASURED["7f+dark"]
        finally:
            widget.hide()

    # ------------------------------------------------------------------ #
    # Cells ▸ Whole patch with Advanced OPEN
    # ------------------------------------------------------------------ #
    #: Advanced-open states are outside the 3.1-screen ceiling already (the
    #: Refine precedent above), so these are held to the heights the page
    #: had before the whole patch began to fill to the walls with a
    #: background marker: Cells ▸ Whole patch, Markers ▸ a run, Advanced
    #: open, before a result (7k, idle and busy) and over candidates (7i,
    #: idle and busy), with two image channels and with one ("1").  Measured
    #: 2026-11-15 offscreen at 312 px on the untouched page; a state may get
    #: shorter, never taller.
    #: 7k+sp (2026-11-24) is new: a Cells whole patch at the never-set
    #: spacing used to refuse; it runs now, and with the derive note holding
    #: the amber slot the spacing note sits in the Advanced help slot.  It
    #: had no earlier height, so it is held under the tallest Advanced-open
    #: whole patch pinned below (7f-whole adv, 995 px).
    CELLS_WHOLE_ADV_BEFORE = {"7k busy": 893, "7k idle": 931,
                              "7i busy": 973, "7i idle": 973,
                              "7k1 busy": 893, "7k1 idle": 931,
                              "7i1 busy": 973, "7i1 idle": 973,
                              "7k+sp busy": 995, "7k+sp idle": 995}
    #: the same states as the page draws them now.  The whole-patch line
    #: (three lines) had replaced Grow by, Wall cost and Growth limit (three
    #: rows), 31 px shorter each; 7k+sp (measured 2026-11-24): 7k plus the
    #: spacing note in the Advanced help slot, 59 px over 7k.  Grow by came
    #: back on a whole patch (owner ruling 2026-11-24; Fill picked, so Wall
    #: cost and Growth limit stay hidden): +26 px each, still 5 px (7k, 7i)
    #: and 10 px (7k+sp) under the heights above; no ceiling raised.
    CELLS_WHOLE_ADV_MEASURED = {"7k busy": 888, "7k idle": 926,
                                "7i busy": 968, "7i idle": 968,
                                "7k1 busy": 888, "7k1 idle": 926,
                                "7i1 busy": 968, "7i1 idle": 968,
                                "7k+sp busy": 947, "7k+sp idle": 985}
    CELLS_WHOLE_ADV_STATES = ("7k", "7i", "7k1", "7i1", "7k+sp")

    def test_add_cells_whole_patch_advanced_open(self, widget):
        """Cells ▸ Whole patch with Advanced open: no state taller than it
        was before the fill to the walls, every number pinned."""
        heights = {}
        for state in self.CELLS_WHOLE_ADV_STATES:
            one = state.endswith("1")
            self._cells_state(widget, state[:-1] if one else state,
                              one_channel=one)
            try:
                widget._add_advanced_toggle.setChecked(True)
                assert not widget._add_advanced_container.isHidden()
                assert not widget._add_cells_adv.isHidden()
                # what a whole patch shows under Advanced: the line that
                # says what fills it, Grow by (the user's pick there too)
                # and Include membrane; on Fill never Wall cost or Growth
                # limit
                assert not widget._add_grow_row.isHidden()
                for row in (widget._add_wall_cost_row,
                            widget._add_growth_limit_row):
                    assert row.isHidden()
                assert not widget._add_whole_fill_label.isHidden()
                assert not widget.add_include_membrane_cb.isHidden()
                if state == "7k+sp":
                    from packages.epicycle_annotator import (
                        add_preview as _AP,
                    )

                    assert _AP.ADD_NOTE_DEFAULT_SPACING_BOXED in (
                        widget._add_advanced_help.text())
                    assert not widget._add_advanced_help.isHidden()
                busy, idle = self._derive_heights(widget)
                heights[state + " busy"] = busy
                heights[state + " idle"] = idle
                print("[ADD-HEIGHT] {} Advanced open: busy {} / idle {} "
                      "px".format(state, busy, idle))
            finally:
                widget._add_advanced_toggle.setChecked(False)
                self._undo_cells_state(widget)
                widget.update_add_ui_state(busy=False, derive_note="",
                                           whole_patch=False)
                widget.hide()
        print("[ADD-HEIGHT] whole patch, Advanced open: {}".format(heights))
        for key, h in heights.items():
            before = self.CELLS_WHOLE_ADV_BEFORE.get(key)
            assert before is not None, (key, h)
            assert h <= before, (
                "Add {} with Advanced open is {} px, taller than the {} px "
                "it was -- trim the content".format(key, h, before))
            assert h == self.CELLS_WHOLE_ADV_MEASURED.get(key), (
                "Add {} with Advanced open moved: {} px, recorded {} -- "
                "record the new number with the reason".format(
                    key, h, self.CELLS_WHOLE_ADV_MEASURED.get(key)))

    # ------------------------------------------------------------------ #
    # Cells ▸ Whole patch with Click markers ("7f-whole")
    # ------------------------------------------------------------------ #
    #: 7f-whole = Cells on a ONE-channel patch (both automatic sources
    #: grey), Whole patch, Click markers, 12 markers, the Markers card open,
    #: S1 and the amber slot at their longest (the amber: the ridge note at
    #: the fill's ceiling).  Measured 2026-11-21 offscreen at 340 px in a
    #: 245 px screen, busy / idle.  With the membrane-only amber shown it
    #: was 752 / 790 px, over the 759.5 px ceiling; that amber is now off on
    #: a whole patch (Click markers are already picked there, and the whole
    #: patch's help line says how they fill), the one row taken off: now
    #: 667 / 705 px (2.88 screens), 54.5 px under the ceiling.
    #: "7f-whole adv" is the same state with Advanced open (outside the
    #: ceiling, as every Advanced-open state; a measured pin, not a
    #: ceiling): 942 / 980 px -- the marker list's rows over 7k's 862 /
    #: 900.  It moved 980 -> 995 (957 / 995 px) when "Update as I click"
    #: came to Cells ▸ Click markers (its row under Advanced, 2026-11-22);
    #: no ceiling was raised.  It moved 995 -> 1036 (998 / 1036 px) on
    #: 2026-11-24: Grow by back on a whole patch (+26 px) and the "Always
    #: update as I click" tick under the live tick (+15 px); 7f-whole with
    #: Advanced closed is unchanged (705), no ceiling was raised.
    #: "7f-whole adv note" (2026-11-24) is 7f-whole adv with the grey line
    #: under "Update as I click" shown at its longest (a click not run as
    #: you click here; the line is a row only in those states): measured
    #: 1023 / 1061 px, +25 px over 7f-whole adv -- every reason of 54-60
    #: characters wraps to two lines at the 312 px viewport ("Preparing
    #: the fill…" is one).  A measured pin, outside the ceiling as every
    #: Advanced-open state; 7f-whole and 7f-whole adv are unchanged, no
    #: ceiling was raised.
    CELLS_WHOLE_CLICKS_MEASURED = {"7f-whole": 705, "7f-whole adv": 1036,
                                   "7f-whole adv note": 1061}

    @staticmethod
    def _longest_live_note():
        from packages.epicycle_annotator import add_preview as AP

        return max((AP.ADD_LIVE_NOTE_OVER.format(n=AP.voxel_limit_words(
                        AP.ADD_LIVE_CELLS_MAX_VOXELS)),
                    AP.ADD_LIVE_NOTE_WALLS, AP.ADD_LIVE_NOTE_GROW,
                    AP.ADD_LIVE_NOTE_FLAT, AP.ADD_LIVE_NOTE_SPACING,
                    AP.ADD_LIVE_NOTE_OFF, AP.ADD_LIVE_NOTE_PREPARING),
                   key=len)

    def _state_7f_whole(self, widget):
        from qtpy.QtWidgets import QApplication

        from packages.epicycle_annotator import add_preview as AP

        widget.resize(340, 350)
        widget.show()
        QApplication.processEvents()
        widget.focus_tab("Add")
        widget.refresh_add_image_combo(["nuc"])
        widget.set_add_preset("cells")
        widget.update_add_ui_state(marker_sources=[
            (AP.ADD_ITEM_NO_RUN, False, ("labels", None)),
            (AP.ADD_ITEM_NUCLEI_GREY, False, ("nuclei_masks", None)),
            (AP.ADD_ITEM_CLICKS, True, ("paint", None))],
            box_active=True, voxel_size_known=True, whole_patch=True,
            single_channel=True, cleared_in_force=True)
        keys = list(widget._add_source_keys)
        widget.add_marker_source_combo.setCurrentIndex(
            keys.index(("paint", None)))
        widget.add_markers_btn.setChecked(True)
        markers = [(65520 + i, (1234, 1234, 1234), i % 3 != 1)
                   for i in range(12)]
        widget.update_add_ui_state(
            click_count=12, click_markers=markers, candidate_ids=[],
            candidate_sizes={}, candidate_edge_ids=set(),
            derive_note=self._longest_cells_amber(),
            status_text=self._longest_cells_s1())
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        return QApplication

    def test_add_cells_whole_click_markers_fit(self, widget):
        """7f-whole (and its Advanced-open twin): the tallest Cells whole
        patch -- Click markers on one channel -- fits the 3.1 ceiling with
        margin; the membrane-only amber is off on a whole patch; both
        numbers pinned."""
        heights = {}
        self._state_7f_whole(widget)
        try:
            assert widget.get_add_marker_source() == ("paint", None)
            assert not widget._add_click_rows.isHidden()
            assert not widget.add_click_list.isHidden()
            assert widget._add_auto_sources_grey()
            assert not widget._add_derive_note.isHidden()
            assert widget._add_derive_note.text() == (
                self._longest_cells_amber())
            assert widget._add_membrane_only_label.isHidden(), (
                "the membrane-only amber is off on a whole patch")
            viewport = self._screen_h(widget)
            busy, idle = self._busy_and_idle(widget)
            heights["7f-whole"] = max(busy, idle)
            print("[ADD-HEIGHT] 7f-whole: busy {} / idle {} px in {} px = "
                  "{:.3f} screens".format(busy, idle, viewport,
                                          max(busy, idle) / viewport))
            width = _slots(widget, "Add")["content"].minimumSizeHint(
            ).width()
            assert width <= self._viewport_w(widget), width
            assert heights["7f-whole"] <= (
                self.ADD_CANDIDATES_SCREENS * viewport - 20), (
                "Add 7f-whole is {} px in a {} px screen -- never raise "
                "3.1".format(heights["7f-whole"], viewport))
            widget._add_advanced_toggle.setChecked(True)
            assert not widget._add_whole_fill_label.isHidden()
            busy, idle = self._busy_and_idle(widget)
            heights["7f-whole adv"] = max(busy, idle)
            print("[ADD-HEIGHT] 7f-whole Advanced open: busy {} / idle {} "
                  "px".format(busy, idle))
            # the line under the live tick, at its longest
            assert widget._add_live_note.isHidden()
            width_before = _slots(widget, "Add")["content"].minimumSizeHint(
            ).width()
            widget.update_add_ui_state(
                live_tick_note=self._longest_live_note())
            assert not widget._add_live_note.isHidden()
            busy, idle = self._busy_and_idle(widget)
            heights["7f-whole adv note"] = max(busy, idle)
            print("[ADD-HEIGHT] 7f-whole Advanced open, the live tick's "
                  "line: busy {} / idle {} px".format(busy, idle))
            # it wraps: the page is no wider with it than without it
            width = _slots(widget, "Add")["content"].minimumSizeHint(
            ).width()
            assert width == width_before, (width, width_before)
        finally:
            widget.update_add_ui_state(live_tick_note="")
            widget._add_advanced_toggle.setChecked(False)
            widget.update_add_ui_state(click_count=0, click_markers=[])
            self._undo_cells_state(widget)
            widget.update_add_ui_state(busy=False, derive_note="",
                                       whole_patch=False)
            widget.hide()
        for key, h in heights.items():
            assert h == self.CELLS_WHOLE_CLICKS_MEASURED[key], (
                "Add {} moved: {} px, recorded {} -- record the new number "
                "with the reason".format(
                    key, h, self.CELLS_WHOLE_CLICKS_MEASURED[key]))

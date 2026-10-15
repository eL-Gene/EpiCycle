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
        if name == "Add":
            # Add's "Accept and refine" lives in the footer
            # summary's slot, shown only while candidates are up -- the one
            # named exception; the two footer buttons are still two
            buttons = [b for b in buttons
                       if b is not widget.add_accept_refine_btn]
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
    MEASURED = {"7a": 645, "7a+": 705, "7b": 674, "7c": 675, "7d": 724,
                "7e": 669, "7f": 739, "7f+dark": 751}
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
        widget.update_add_ui_state(status_text=self._longest_s1(single),
                                   busy=True, stop_enabled=True,
                                   stage_text="Filling the empty space...")
        # offscreen Qt settles the scroll area a pass or two late, and the
        # first panel built in a process can come up wider than the dock
        widget.resize(340, 350)
        for _ in range(3):
            QApplication.processEvents()
        return QApplication

    @staticmethod
    def _longest_s1(single):
        """The longest S1 sentence the state can reach: over candidates,
        the longest of the sentences that can stand above a result; the
        single-channel state (7c) at the longest sentence a Cells page on
        the clicks source can show before a result -- it was
        ADD_S1_NO_VOXEL_SIZE until retired it."""
        from packages.epicycle_annotator import add_preview as AP

        if single:
            return max((AP.over_limit_line((200, 200, 200)),
                        AP.ADD_S1_NO_MARKERS, AP.ADD_S1_PICK_WALL_CHANNEL,
                        AP.ADD_S1_CLICK_ON_KEPT.format(id=123456),
                        AP.ADD_S1_CLICKS_DROPPED.format(n=12),
                        AP.ADD_S1_BUSY_REFUSED, AP.ADD_S1_FAILED,
                        AP.ADD_S1_ALL_FILTERED, AP.ADD_S1_INPUTS_CHANGED),
                       key=len)
        return max((AP.ADD_S1_TAKEN, AP.ADD_S1_FAILED, AP.ADD_S1_STOPPED,
                    AP.ADD_S1_INPUTS_CHANGED, AP.ADD_S1_BUSY_REFUSED,
                    AP.ADD_S1_TICK_ONE,
                    AP.found_line(12, k_edge=2, k_small=2, k_both=1)),
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

        self._state_7f(widget)
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

"""The Reshape tab's 13 states, and the footer's two-value secondary.

``update_split_ui_state`` had NO test at all before the panel rework: 37
call sites in the annotator and zero references in ``tests/``.  This file
drives every state and asserts what is on screen, what is
enabled, and what the two footer buttons say.

**The footer's secondary is keyed on ONE predicate: is the target punched
out.**  ``_start_split`` zeroes the target in the shared volume, so from
that moment the secondary's job is to put the cell back -- "Discard".
Only in idle, where there is nothing to undo, does it read "Cancel"
.  Never "Reject", never "Reject All".

Pattern 3 (offscreen QApplication), plus per-method source
assertions for the annotator's push sites (there is no other way to check
that BOTH staging paths push ``review_active`` -- the manual one stages
without ever calling the guided setter).
"""

from __future__ import annotations

import inspect
import re

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


def _ann():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


def _widget_class():
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    return AdvancedFeaturesWidget


# --------------------------------------------------------------------------- #
# the states, as the orchestrator pushes them
# --------------------------------------------------------------------------- #

def _idle(w):
    w.update_split_ui_state(
        seed_count=0, seed_ids=[], split_active=False, edt_enabled=False,
        start_enabled=True, next_seed_enabled=False, undo_seed_enabled=False,
        refine_seeds_enabled=False, preview_enabled=False,
        accept_enabled=False, reject_enabled=False, reject_text="Cancel",
        refine_further_enabled=False, fragment_refine_active=False,
        detect_enabled=False, detect_visible=False, review_active=False,
        status_text="Select a label, then Start Split.",
    )


def _started(w):
    w.update_split_ui_state(
        seed_count=1, seed_ids=[9], active_seed_id=9, split_active=True,
        edt_enabled=True, start_enabled=False, next_seed_enabled=True,
        undo_seed_enabled=False, refine_seeds_enabled=False,
        preview_enabled=False, accept_enabled=False, reject_enabled=True,
        reject_text="Discard", detect_enabled=True, detect_visible=False,
        review_active=False, status_text="Painting seed 1 (label 9).",
    )


def _solving(w):
    w.update_split_guided_ui(busy=True, busy_progress=(0, 0))
    w.update_split_ui_state(
        preview_enabled=False, accept_enabled=False, reject_enabled=True,
        reject_text="Discard", status_text="Working.",
    )


def _review(w, pieces=(11, 12, 13)):
    w.update_split_guided_ui(busy=False)
    w.update_split_guided_ui(
        pieces=list(pieces),
        piece_sizes={p: 4000 for p in pieces},
        caption="unticked pieces stay part of cell 149",
    )
    w.update_split_ui_state(
        preview_enabled=True, accept_enabled=True, reject_enabled=True,
        reject_text="Discard", review_active=True,
        status_text="{} pieces. Tick the ones you want, then Accept.".format(
            len(pieces)),
    )


# --------------------------------------------------------------------------- #
# I-5 -- the footer's secondary takes exactly two values
# --------------------------------------------------------------------------- #
class TestTheSecondaryIsKeyedOnThePunchOut:
    def test_idle_is_the_only_state_that_says_cancel(self, widget):
        _idle(widget)
        assert widget.split_reject_btn.text() == "Cancel"
        assert widget.split_reject_btn.isEnabled() is False
        for step in (_started, _solving, _review):
            step(widget)
            assert widget.split_reject_btn.text() == "Discard", (
                "the cell is out of the volume; the button that puts it "
                "back must say so"
            )
            assert widget.split_reject_btn.isEnabled() is True

    def test_the_annotator_only_ever_writes_those_two_words(self):
        src = inspect.getsource(_ann())
        found = set(re.findall(r'reject_text=\s*"([^"]*)"', src))
        assert found <= {"Cancel", "Discard"}, (
            "the footer secondary takes two values; found {}".format(
                sorted(found))
        )

    @pytest.mark.parametrize("method", [
        "_start_split", "_next_seed", "_undo_last_seed",
        "_apply_seed_threshold", "_refine_seeds", "_on_reshape_cancel",
        "_on_reshape_finished",
        "_preview_split", "_apply_reshape_result",
        "_enter_fragment_refine_mode",
    ])
    def test_every_punched_out_state_says_discard(self, method):
        src = inspect.getsource(getattr(_ann(), method))
        labels = set(re.findall(r'reject_text=\s*"([^"]*)"', src))
        assert labels == {"Discard"}, (
            "{} runs with the target punched out of the volume; it wrote "
            "{}".format(method, sorted(labels))
        )

    @pytest.mark.parametrize("method", ["_accept_split", "_reject_split"])
    def test_the_terminal_resets_go_back_to_cancel(self, method):
        """Both end with the cell either written or restored, so the panel
        is back in idle and there is nothing left to undo."""
        src = inspect.getsource(getattr(_ann(), method))
        labels = set(re.findall(r'reject_text=\s*"([^"]*)"', src))
        assert labels == {"Cancel"}, (
            "{} wrote {}".format(method, sorted(labels))
        )

    def test_reject_all_is_gone(self):
        src = inspect.getsource(_ann())
        assert '"Reject All"' not in src


# --------------------------------------------------------------------------- #
# I-2 -- review_active drives the S5 swap, from BOTH staging sites
# --------------------------------------------------------------------------- #
class TestReviewSwap:
    def test_the_result_section_is_hidden_until_review(self, widget):
        _idle(widget)
        assert widget._split_result_container.isVisibleTo(widget) is False
        _started(widget)
        assert widget._split_result_container.isVisibleTo(widget) is False
        _review(widget)
        assert widget._split_result_container.isVisibleTo(widget) is True
        assert widget.split_piece_list.isVisibleTo(widget) is True

    def test_leaving_review_takes_the_result_away(self, widget):
        _review(widget)
        widget.update_split_ui_state(review_active=False)
        assert widget._split_result_container.isVisibleTo(widget) is False

    def test_review_LEAVES_sources_open(self, widget):
        """INVERTED in Brick U1 (UX round 1.1/1.4).

        Review used to collapse SOURCES, which is the one line that made
        the flow a line instead of a loop: after a Preview the only
        thing on offer was "Preview again" with the same markers.  A
        running SOLVE still collapses it -- you must not re-tick markers
        a worker is using -- and that is asserted below."""
        widget.split_guided_summary_btn.setChecked(True)
        _review(widget)
        assert widget.split_guided_summary_btn.isChecked() is True
        assert widget.split_guided_summary_btn.isEnabled() is True
        assert widget._split_guided_container.isVisibleTo(widget) is True

    def test_a_running_solve_still_collapses_sources(self, widget):
        """A-L11: by DISABLING the chevron, never `setChecked(False)` --
        the container is driven by that button's `toggled` signal, so
        unchecking it would lose the user's disclosure state."""
        widget.split_guided_summary_btn.setChecked(True)
        _solving(widget)
        assert widget.split_guided_summary_btn.isChecked() is True
        assert widget.split_guided_summary_btn.isEnabled() is False
        assert widget._split_guided_container.isVisibleTo(widget) is False
        widget.update_split_guided_ui(busy=False)
        assert widget.split_guided_summary_btn.isEnabled() is True
        assert widget._split_guided_container.isVisibleTo(widget) is True

    def test_preview_reads_preview_again_in_review(self, widget):
        _started(widget)
        assert widget.split_preview_btn.text() == "Preview"
        _review(widget)
        assert widget.split_preview_btn.text() == "Preview again"
        widget.update_split_ui_state(review_active=False)
        assert widget.split_preview_btn.text() == "Preview"

    def test_BOTH_staging_sites_push_review_active(self):
        """The manual path stages without ever calling the guided setter,
        so inferring "review" from len(pieces) > 0 would silently fail it."""
        for method in ("_preview_split", "_apply_reshape_result"):
            src = inspect.getsource(getattr(_ann(), method))
            assert "review_active=True" in src, (
                "{} stages a result without telling the panel".format(method)
            )

    def test_every_terminal_reset_pushes_review_active_False(self):
        for method in ("_accept_split", "_reject_split"):
            src = inspect.getsource(getattr(_ann(), method))
            assert "review_active=False" in src, (
                "{} leaves the RESULT section on screen".format(method)
            )


# --------------------------------------------------------------------------- #
# state #10 -- fragment refine
# --------------------------------------------------------------------------- #
class TestFragmentRefine:
    def test_accept_text_retitles_the_primary(self, widget):
        _review(widget)
        assert widget.split_accept_btn.text() == "Accept (3)"
        widget.update_split_ui_state(accept_text="Accept all (3)")
        assert widget.split_accept_btn.text() == "Accept all (3)"

    def test_the_fragment_list_replaces_the_piece_list(self, widget):
        _review(widget)
        widget.update_split_guided_ui(pieces=[], caption="Refining 3 fragments")
        widget.update_split_ui_state(
            fragment_refine_active=True, fragment_ids=[11, 12, 13],
            active_fragment_id=11, accept_text="Accept all (3)",
            reject_text="Discard", review_active=True,
        )
        assert widget.split_piece_list.count() == 0
        assert widget.split_fragment_list.count() == 3
        assert widget.split_fragment_list.isVisibleTo(widget) is True
        assert widget.split_piece_list.isVisibleTo(widget) is False

    def test_the_annotator_clears_the_checklist_on_entry(self):
        """Nothing cleared it before, so S5 would show both lists at once."""
        src = inspect.getsource(_ann()._enter_fragment_refine_mode)
        assert "update_split_guided_ui" in src
        assert "pieces=[]" in src
        assert "accept_text=" in src


# --------------------------------------------------------------------------- #
# S1 status + the footer summary line
# --------------------------------------------------------------------------- #
class TestTheProseMatchesTheButtons:
    """Status lines that name a button must name the button that exists.

    The panel's secondaries now read Cancel / Discard (and "Done" in Add's
    REVIEW).  Any status text still saying "Reject", "Reject All" or
    "press Cancel" is instructing the user to press something that is not
    on screen.

    The annotator's OWN interpolation buttons still read Accept/Reject --
    they are not part of this panel -- so prose about *them* is correct and
    is deliberately not swept.
    """

    #: methods whose status text talks about a panel button.
    PANEL_PROSE = (
        "_preview_split", "_on_reshape_cancel", "_on_reshape_error",
        "_on_reshape_finished", "_enter_fragment_refine_mode",
        "_accept_smooth", "_reject_smooth", "_enter_interp_refine_mode",
        "_clear_draw_edits",
    )

    @pytest.mark.parametrize("method", PANEL_PROSE)
    def test_no_status_line_names_a_button_that_is_gone(self, method):
        src = inspect.getsource(getattr(_ann(), method))
        # only the STRING literals, so method names like _reject_smooth and
        # comments about the Reject path stay legal.
        strings = re.findall(r'"([^"\n]{12,})"', src)
        for s in strings:
            low = s.lower()
            assert "reject all" not in low, (method, s)
            assert "or reject" not in low, (method, s)
            assert "press cancel" not in low, (method, s)

    def test_stopping_a_solve_does_not_claim_nothing_changed(self):
        """Stop abandons the worker and LEAVES THE CELL PUNCHED OUT; only
        the footer's secondary puts it back (Stop != Discard)."""
        src = inspect.getsource(_ann()._on_reshape_cancel)
        assert "nothing was changed" not in src, (
            "the cell is still held out of the volume at this point"
        )
        assert "Discard" in src

    def test_an_errored_solve_does_not_call_the_cell_untouched(self):
        src = inspect.getsource(_ann()._on_reshape_error)
        assert "cell is untouched" not in src
        assert "Discard" in src

    def test_fragment_refine_says_accept_all_and_discard(self):
        src = inspect.getsource(_ann()._enter_fragment_refine_mode)
        assert "Accept all" in src
        assert "Reject All" not in src


class TestDisclosureStateBelongsToTheUser:
    """The live screenshots show ADVANCED still open in review, after being
    opened in an earlier state.  That is the user's own disclosure state
    being preserved, not a reset bug: nothing but the user's click writes
    `split_advanced_btn` or `_split_advanced_container` after construction.

    SOURCES is the one that IS driven by state -- and even that is only
    COLLAPSED (by disabling its chevron), never unchecked, so it comes back
    exactly as the user left it (A-L11).
    """

    def test_advanced_stays_open_through_every_state(self, widget):
        widget.split_advanced_btn.setChecked(True)
        assert widget._split_advanced_container.isVisibleTo(widget) is True
        for step in (_started, _solving, _review, _idle):
            step(widget)
            assert widget.split_advanced_btn.isChecked() is True, (
                "a stage change closed the user's Advanced disclosure"
            )
            assert widget._split_advanced_container.isVisibleTo(widget) is True
            assert widget.split_advanced_btn.isEnabled() is True

    def test_sources_is_collapsed_but_remembered(self, widget):
        widget.split_guided_summary_btn.setChecked(True)
        _solving(widget)
        assert widget._split_guided_container.isVisibleTo(widget) is False
        assert widget.split_guided_summary_btn.isChecked() is True
        widget.update_split_guided_ui(busy=False)
        widget.update_split_ui_state(review_active=False)
        assert widget._split_guided_container.isVisibleTo(widget) is True

    def test_only_the_user_writes_the_advanced_disclosure(self):
        """A source pin: the state setters must never reach it."""
        for name in ("update_split_ui_state", "update_split_guided_ui",
                     "_clear_split_result_rows", "clear_split_guided_ui",
                     "_refresh_split_stage"):
            src = inspect.getsource(
                getattr(_widget_class(), name))
            assert "split_advanced_btn" not in src, name
            assert "_split_advanced_container" not in src, name


class TestBirthState:
    """What the panel says BEFORE anything pushes a state into it.

    Nothing pushes idle when the dock opens, so whatever the constructor
    wrote is what the user reads first.
    """

    def test_the_secondary_reads_Cancel_at_birth(self, widget):
        assert widget.split_reject_btn.text() == "Cancel", (
            "the dock opens on a button the two-value rule does not allow"
        )

    def test_the_footer_line_does_not_promise_a_staged_result(self, widget):
        assert widget._split_footer_summary.text() == "Nothing is staged yet."

    def test_the_primary_and_the_target_button_agree_with_it(self, widget):
        assert widget.split_accept_btn.text() == "Accept"
        assert widget.split_start_btn.text() == "Start"
        assert widget.split_preview_btn.text() == "Preview"


class TestAcceptTextIsResetWithTheReview:
    """Fragment refine writes "Accept all (n)"; nothing used to take it
    back, so it leaked into the NEXT session and even into a manual
    2-piece preview."""

    def test_leaving_review_resets_the_primary(self, widget):
        _review(widget)
        widget.update_split_ui_state(accept_text="Accept all (3)")
        assert widget.split_accept_btn.text() == "Accept all (3)"
        widget.update_split_ui_state(review_active=False)
        assert widget.split_accept_btn.text() == "Accept"

    def test_a_later_session_counts_its_own_pieces(self, widget):
        # fragment refine -> accept all -> the session ends
        _review(widget)
        widget.update_split_ui_state(
            fragment_refine_active=True, fragment_ids=[11, 12, 13],
            accept_text="Accept all (3)", review_active=True,
        )
        _idle(widget)
        assert widget.split_accept_btn.text() == "Accept"
        # a NEW, manual, two-piece preview
        _started(widget)
        widget.update_split_guided_ui(
            pieces=[21, 22], piece_sizes={21: 500, 22: 480})
        assert widget.split_accept_btn.text() == "Accept (2)", (
            "the primary is still counting the previous session's fragments"
        )


class TestTheTargetButton:
    """D1/D2: one object, two labels, and it is never live mid-session."""

    def test_it_reads_Start_in_idle(self, widget):
        _idle(widget)
        assert widget.split_start_btn.text() == "Start"

    def test_it_reads_Change_once_the_cell_is_held_out(self, widget):
        _started(widget)
        assert widget.split_start_btn.text() == "Change"
        assert widget.split_start_btn.isEnabled() is False, (
            "re-targeting mid-session would abandon an unrestored hole"
        )

    def test_it_goes_back_to_Start_when_the_session_ends(self, widget):
        _started(widget)
        _idle(widget)
        assert widget.split_start_btn.text() == "Start"
        assert widget.split_start_btn.isEnabled() is True

    def test_its_tooltip_says_the_cell_will_be_hidden(self, widget):
        tip = widget.split_start_btn.toolTip().lower()
        assert "hidden" in tip


class TestBlockedReasonReachesTheStatusLine:
    """It was a DEAD parameter: defined, handled, and called by nothing."""

    def test_the_reason_lands_in_S1(self, widget):
        widget.update_split_guided_ui(
            blocked_reason="Waiting for Save curation to finish.")
        assert "Save curation" in widget.split_status_label.text()

    def test_the_reason_also_explains_the_disabled_Accept(self, widget):
        widget.update_split_guided_ui(blocked_reason="Training is running.")
        assert "Training" in widget.split_accept_btn.toolTip()

    def test_clearing_it_restores_the_real_Accept_tooltip(self, widget):
        widget.update_split_guided_ui(blocked_reason="Training is running.")
        widget.update_split_guided_ui(blocked_reason="")
        tip = widget.split_accept_btn.toolTip().lower()
        assert "ticked" in tip and "never deleted" in tip

    def test_it_no_longer_rewrites_the_target_buttons_tooltip(self, widget):
        """That tooltip described seed painting on a button that now says
        "Start"/"Change"."""
        widget.update_split_guided_ui(blocked_reason="Training is running.")
        assert "Training" not in widget.split_start_btn.toolTip()


class TestStatusSurfaces:
    def test_status_text_lands_in_S1(self, widget):
        widget.update_split_ui_state(status_text="Cell 149 is ready.")
        assert widget.split_status_label.text() == "Cell 149 is ready."
        node, inside = widget.split_status_label, False
        while node is not None:
            if node is widget._split_s1_container:
                inside = True
                break
            node = node.parentWidget()
        assert inside

    def test_busy_text_writes_to_the_same_surface(self, widget):
        """It has always had two writers; they must not diverge."""
        widget.update_split_guided_ui(busy_text="finding seeds (2 of 4)")
        assert "finding seeds" in widget.split_status_label.text()

    def test_the_footer_summary_says_what_the_two_buttons_do(self, widget):
        _idle(widget)
        idle_text = widget._split_footer_summary.text()
        _review(widget)
        review_text = widget._split_footer_summary.text()
        assert idle_text != review_text
        assert "Discard" in review_text
        assert review_text, "the footer's one line must not be empty"

    def test_the_state_dot_tracks_the_stage(self, widget):
        seen = set()
        for step in (_idle, _started, _solving, _review):
            step(widget)
            seen.add(widget._split_status_dot.styleSheet())
        assert len(seen) >= 3, (
            "the dot is the state's colour; it cannot be one colour for "
            "every state"
        )


# --------------------------------------------------------------------------- #
# Brick 2a -- states 5a-5d, the marker states
# --------------------------------------------------------------------------- #
def _finding(w):
    """5a: finding markers."""
    w.update_split_ui_state(
        split_active=True, marker_busy=True, preview_enabled=False,
        accept_enabled=False, reject_enabled=True, reject_text="Discard",
        status_text="Cell 149 - finding markers...",
    )


def _markers_found(w, ids=(1, 2, 3)):
    """5b: markers found."""
    w.update_split_ui_state(
        marker_busy=False, marker_ids=list(ids),
        marker_stats={i: 900 for i in ids},
        preview_enabled=True, accept_enabled=False, reject_enabled=True,
        reject_text="Discard",
        status_text="Cell 149 - {} markers found, {} ticked".format(
            len(ids), len(ids)),
    )


class TestTheMarkerStates:
    def test_5a_finding_does_not_collapse_sources(self, widget):
        """The solve's busy state collapses S3; a Find must not -- it would
        hide the button that started it and the list it is about to fill."""
        widget.split_guided_summary_btn.setChecked(True)
        _started(widget)
        _finding(widget)
        assert widget.split_guided_summary_btn.isEnabled() is True
        assert widget._split_guided_container.isVisibleTo(widget) is True
        assert widget.split_find_markers_btn.isEnabled() is False
        assert "Finding" in widget.split_marker_count_label.text()

    def test_5a_still_offers_Discard_because_the_object_is_punched_out(
            self, widget):
        _started(widget)
        _finding(widget)
        assert widget.split_reject_btn.text() == "Discard"
        assert widget.split_accept_btn.isEnabled() is False

    def test_5b_lists_the_markers_and_enables_Preview(self, widget):
        _started(widget)
        _markers_found(widget)
        assert widget.split_marker_list.count() == 3
        assert widget.get_checked_marker_ids() == [1, 2, 3]
        assert widget.split_preview_btn.isEnabled() is True
        assert widget.split_accept_btn.isEnabled() is False

    def test_5c_a_stale_list_clears_and_says_so_on_the_count_line(self, widget):
        _started(widget)
        _markers_found(widget)
        before = widget.split_status_label.text()
        widget.split_seed_h_sb.setValue(0.42)
        assert widget.split_marker_list.count() == 0
        assert "Find again" in widget.split_marker_count_label.text()
        assert widget.split_status_label.text() == before, (
            "the stale sentence belongs on the count line, not on S1"
        )

    def test_5d_a_refusal_shows_its_reason_and_keeps_Preview_alive(self, widget):
        _started(widget)
        widget.update_split_ui_state(
            marker_busy=False, marker_ids=[],
            marker_note="No markers found in this object",
            preview_enabled=True, reject_enabled=True, reject_text="Discard",
            status_text="Object 149 - no markers found",
        )
        assert widget.split_marker_list.count() == 0
        assert "No markers found" in widget.split_marker_count_label.text()
        assert widget.split_preview_btn.isEnabled() is True

    def test_the_review_KEEPS_the_marker_list(self, widget):
        """INVERTED with inv. 9 (UX round 1.4).  Both lists are on
        screen; what holds instead is that they cannot be confused --
        see `test_panel_skeleton.py`."""
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        _started(widget)
        _markers_found(widget)
        assert widget.split_marker_list.isVisibleTo(widget) is True
        _review(widget)
        assert widget.split_marker_list.isVisibleTo(widget) is True
        assert widget.split_piece_list.isVisibleTo(widget) is True


class TestClearResetsTheCherryPick:
    def test_clear_split_guided_ui_resets_keep_seed_ids_to_None(self, widget):
        """A Discard on cell 139 must not pass stale ids to cell 140."""
        _markers_found(widget)
        assert widget.get_checked_marker_ids() == [1, 2, 3]
        widget.clear_split_guided_ui()
        assert widget.get_checked_marker_ids() is None

    def test_the_annotator_clears_it_on_every_terminal_path(self):
        import inspect

        for method in ("_teardown_reshape_for_switch",):
            src = inspect.getsource(getattr(_ann(), method))
            assert "clear_split_guided_ui" in src, method

    def test_clear_also_lets_the_busy_state_go(self, widget):
        widget.update_split_ui_state(marker_busy=True)
        widget.clear_split_guided_ui()
        assert widget.split_find_markers_btn.isEnabled() is True
        assert widget._split_busy_container.isVisibleTo(widget) is False

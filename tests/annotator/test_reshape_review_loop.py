"""Brick U1 -- the loop.

The shipped Reshape tab was a LINE: Start, find markers, Preview -- and
then the Markers section collapsed and the only thing on offer was
"Preview again" with the same markers.  Changing one meant discarding the
whole session.

Two halves are tested here, because un-hiding S3 is necessary but not
sufficient:

* the PANEL no longer collapses Sources at review (a running solve still
  does), and every cut knob leaves review without clearing the markers;
* the ORCHESTRATOR snaps the Refine ROI back to the MARKERS whenever
  anything in S3 changes, so the marker bookkeeping can never be fed a
  partition.  Without the snap-back `_marker_rows_from_refine` would list
  pieces as markers, the recipe would hand the engine the whole previous
  partition as seeds, and Undo would erase a piece.

Qt-free fakes for the orchestrator half, the offscreen panel for the
panel half -- the same split the Brick 2 suites use.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")


# --------------------------------------------------------------------------- #
# panel half
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


def _review(w, pieces=(11, 12, 13)):
    w.update_split_guided_ui(
        busy=False, pieces=list(pieces),
        piece_sizes={p: 4000 for p in pieces},
        caption="unticked pieces stay part of object 149",
    )
    w.update_split_ui_state(review_active=True, accept_enabled=True)


class TestReviewLeavesSourcesOpen:
    def test_review_no_longer_collapses_sources(self, widget):
        """ONE line hid the markers, and it is gone."""
        widget.split_guided_summary_btn.setChecked(True)
        widget.update_split_ui_state(split_active=True)
        _review(widget)
        assert widget.split_guided_summary_btn.isEnabled() is True
        assert widget._split_guided_container.isVisibleTo(widget) is True

    def test_a_running_solve_still_collapses_sources(self, widget):
        """You must not re-tick markers a worker is using."""
        widget.split_guided_summary_btn.setChecked(True)
        widget.update_split_guided_ui(busy=True)
        assert widget.split_guided_summary_btn.isEnabled() is False
        assert widget._split_guided_container.isVisibleTo(widget) is False
        widget.update_split_guided_ui(busy=False)
        assert widget.split_guided_summary_btn.isEnabled() is True

    def test_the_marker_list_survives_the_review(self, widget):
        widget.split_guided_summary_btn.setChecked(True)
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        _review(widget)
        assert widget.split_marker_list.isVisibleTo(widget) is True
        assert widget.split_piece_list.isVisibleTo(widget) is True


class TestARunningSolveOwnsTheMarkers:
    """Fix pass M1: says a running solve locks the marker work, and
    only half of it was locked.

    Collapsing SOURCES hides the PICKERS.  The marker list, the All /
    None chips, Add marker, Undo, Edit markers and the shape finder live
    in the SESSION block, outside that disclosure and still fully live
    -- so you could untick a marker, or paint and Add another, while the
    worker was mid-solve on the set you just changed.
    """

    def _busy(self, w):
        w.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3],
                                next_seed_enabled=True)
        w.update_split_guided_ui(busy=True)

    def test_the_marker_list_goes_dead(self, widget):
        self._busy(widget)
        assert widget.split_marker_list.isEnabled() is False
        assert widget._split_paint_rows_container.isEnabled() is False

    def test_the_all_and_none_chips_go_dead(self, widget):
        self._busy(widget)
        assert widget.split_marker_all_btn.isEnabled() is False
        assert widget.split_marker_none_btn.isEnabled() is False

    def test_add_marker_cannot_be_pressed(self, widget):
        fired = []
        widget.set_split_callbacks(on_next_seed=lambda: fired.append(1))
        self._busy(widget)
        widget.split_next_seed_btn.click()
        assert fired == [], "Add marker fired while the solve was running"

    def test_it_all_comes_back_when_the_answer_arrives(self, widget):
        self._busy(widget)
        widget.update_split_guided_ui(busy=False)
        assert widget.split_marker_list.isEnabled() is True
        assert widget.split_marker_all_btn.isEnabled() is True
        assert widget.split_next_seed_btn.isEnabled() is True
        assert widget._split_paint_rows_container.isEnabled() is True

    def test_a_button_that_was_ALREADY_off_stays_off(self, widget):
        """The lock restores what it found, not "everything on" -- Undo
        is disabled below two markers and Edit markers has nothing to do
        before there is a preview."""
        widget.update_split_ui_state(split_active=True, marker_ids=[1],
                                     undo_seed_enabled=False,
                                     next_seed_enabled=True)
        widget.update_split_guided_ui(busy=True)
        widget.update_split_guided_ui(busy=False)
        assert widget.split_undo_seed_btn.isEnabled() is False
        assert widget.split_next_seed_btn.isEnabled() is True


class TestCutKnobsLeaveReviewWithoutClearingTheMarkers:
    def test_every_cut_knob_is_wired(self, widget):
        """Set equality against the tuple: a knob added to "How to cut"
        without being added here leaves a stale result with Accept live."""
        for attr in widget._CUT_KNOBS:
            assert hasattr(widget, attr), attr

    def test_a_wall_kind_change_fires_the_callback(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        fired = []
        widget.set_split_callbacks(on_cut_knob_changed=lambda: fired.append(1))
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        _review(widget)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        assert fired, "a wall-kind change must leave review"

    def test_it_does_NOT_clear_the_marker_list(self, widget):
        """The markers are still the markers."""
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        _review(widget)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        assert widget.split_marker_list.count() == 3
        assert widget.get_checked_marker_ids() == [1, 2, 3]

    def test_a_MARKER_knob_still_clears_the_list(self, widget):
        """The other half of the rule, unchanged from Brick 2a."""
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_seed_detector_combo.setCurrentIndex(
            1 if widget.split_seed_detector_combo.currentIndex() == 0 else 0)
        assert widget.split_marker_list.count() == 0
        assert M is not None

    def test_nothing_fires_when_there_is_no_review(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        fired = []
        widget.set_split_callbacks(on_cut_knob_changed=lambda: fired.append(1))
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        assert fired == []


class TestAcceptSaysWhyItIsGrey:
    def test_the_reason_lands_on_accepts_tooltip(self, widget):
        widget.update_split_ui_state(split_active=True, accept_enabled=True)
        widget.set_accept_reason("Markers changed - press Preview to cut again.")
        assert "Markers changed" in widget.split_accept_btn.toolTip()

    def test_re_enabling_accept_takes_the_reason_away(self, widget):
        widget.set_accept_reason("Cut settings changed.")
        widget.update_split_ui_state(accept_enabled=True)
        assert "Cut settings changed" not in widget.split_accept_btn.toolTip()
        assert widget.split_accept_btn.toolTip().strip()

    def test_the_orchestrator_pushes_it_when_it_leaves_review(self):
        import inspect

        src = inspect.getsource(_K()._leave_reshape_review)
        assert "set_accept_reason(reason)" in src


class TestTheFooterHintCarriesTheReasonToo:
    """Fix pass H3: asks for the SAME sentence in three places --
    S1, the footer hint and Accept's tooltip.  The footer hint was keyed
    on the reject button's TEXT alone, and a snap-back does not touch
    that button, so under a greyed-out Accept the footer went on saying
    "Accept keeps the ticked pieces" -- an instruction for a button that
    could not be pressed, and the one line sitting right beside it.
    """

    def test_the_reason_replaces_the_staged_sentence(self, widget):
        widget.update_split_ui_state(split_active=True, reject_text="Discard")
        widget.set_footer_reason("Markers changed - press Preview to cut again.")
        assert widget._split_footer_summary.text() == (
            "Markers changed - press Preview to cut again.")

    def test_re_enabling_accept_takes_it_away(self, widget):
        widget.update_split_ui_state(split_active=True, reject_text="Discard")
        widget.set_footer_reason("Cut settings changed.")
        widget.update_split_ui_state(accept_enabled=True)
        assert widget._split_footer_summary.text() == (
            "Accept keeps the ticked pieces. Discard restores the object.")

    def test_idle_still_wins(self, widget):
        """A reason left over from a session must not survive into idle,
        where there is nothing staged to explain."""
        widget.set_footer_reason("Markers changed.")
        widget.update_split_ui_state(reject_text="Cancel")
        assert widget._split_footer_summary.text() == "Nothing is staged yet."

    def test_the_orchestrator_pushes_it_when_it_leaves_review(self):
        import inspect

        src = inspect.getsource(_K()._leave_reshape_review)
        assert "set_footer_reason(reason)" in src


class TestAMarkerTickLeavesReview:
    def test_a_tick_fires_the_callback_in_review(self, widget):
        from qtpy.QtCore import Qt

        fired = []
        widget.set_split_callbacks(
            on_marker_check_changed=lambda: fired.append(1))
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        _review(widget)
        widget.split_marker_list.item(1).setCheckState(Qt.Unchecked)
        assert fired, "a marker tick must leave review"

    def test_a_tick_outside_review_is_ALSO_reported_now(self, widget):
        """INVERTED by U2b It used to assert `fired == []`,
        because leaving review was the only thing a tick had to do.
        Since an unticked marker comes OFF the object the tick also has
        to lift its voxels -- which is just as necessary with no preview
        on screen -- so the callback is unconditional.  What is still
        review-only is the `leaving review` diagnostic line."""
        from qtpy.QtCore import Qt

        fired = []
        widget.set_split_callbacks(
            on_marker_check_changed=lambda: fired.append(1))
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_list.item(1).setCheckState(Qt.Unchecked)
        assert fired == [1]
        assert widget._split_review_active is False

    def test_a_PIECE_tick_does_not_leave_review(self, widget):
        """Row 1: a piece tick is a redraw, never a re-solve."""
        from qtpy.QtCore import Qt

        fired = []
        widget.set_split_callbacks(
            on_marker_check_changed=lambda: fired.append(1))
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        _review(widget)
        widget.split_piece_list.item(0).setCheckState(Qt.Unchecked)
        assert fired == []
        assert widget._split_review_active is True


# --------------------------------------------------------------------------- #
# orchestrator half
# --------------------------------------------------------------------------- #
def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


class FakeLayer:
    def __init__(self, data):
        self.data = data
        self.refreshed = 0
        self.selected_label = 0
        self.mode = "paint"

    def refresh(self):
        self.refreshed += 1


class FakeAdv:
    def __init__(self, checked=None):
        self._checked = checked
        self.calls = []
        self.reasons = []
        self.footer_reasons = []

    def get_checked_marker_ids(self):
        return self._checked

    def set_accept_reason(self, reason=""):
        self.reasons.append(reason)

    def set_footer_reason(self, reason=""):
        self.footer_reasons.append(reason)

    def update_split_ui_state(self, **kw):
        self.calls.append(kw)

    def get_split_params(self):
        return {"seeds_from": 0}


class Fake:
    """Only what `_leave_reshape_review` and friends touch."""

    def __init__(self, roi_markers, roi_pieces, target=7):
        shape = roi_markers.shape
        self._split_roi = (slice(0, shape[0]), slice(0, shape[1]),
                           slice(0, shape[2]))
        self._refine_layer = FakeLayer(roi_pieces.copy())
        self._refine_pre_preview = roi_markers.copy()
        self._split_label_id = target
        self._split_running = True
        self._split_previewed = True
        self._split_painted_marker_ids = [101, 102]
        self._split_seed_ids = [200, 201, 202]
        self._reshape_result = object()
        self._reshape_final_ids = {1: 200, 2: 201}
        self._reshape_flags = {200: "x"}
        self._reshape_arbiter_merged = True
        self._reshape_override = True
        self._adv = FakeAdv()
        self.ui_calls = []
        self.subviewer_refreshes = 0

    def _get_advanced_features_widget(self):
        return self._adv

    def _update_split_ui(self, **kw):
        self.ui_calls.append(kw)

    def _refresh_refine_in_subviewers(self):
        self.subviewer_refreshes += 1


def _markers_and_pieces():
    markers = np.zeros((4, 4, 4), np.int32)
    markers[0, 0, 0] = 101
    markers[3, 3, 3] = 102
    pieces = np.full((4, 4, 4), 200, np.int32)
    pieces[2:, :, :] = 201
    return markers, pieces


#: Brick U3: the snap-back drops the wall picture, UNGUARDED -- a
#: getattr dance there would let a future exit skip it silently.
#: So every fake that drives it binds the REAL dropper; a stub
#: would pass this file by not having the wire.
_ALWAYS_BOUND = (
    "_drop_walls_picture", "_show_walls_picture",
    "_walls_display_crop", "_remove_temp_layer",
)


def _bind(fake, *names):
    K = _K()
    for n in tuple(names) + _ALWAYS_BOUND:
        setattr(fake, n, getattr(K, n).__get__(fake, Fake))
    return fake


class TestTheSnapBackSaysWhyOnTheFooter:
    def test_the_same_sentence_reaches_the_footer(self):
        m, pieces = _markers_and_pieces()
        fake = Fake(m, pieces)
        _bind(fake, "_leave_reshape_review", "_marker_rows_from_refine")
        fake._leave_reshape_review("Markers changed - press Preview again.")
        assert fake._adv.footer_reasons == [
            "Markers changed - press Preview again."]
        assert fake._adv.reasons == fake._adv.footer_reasons, (
            "S1, the footer and Accept must carry the SAME sentence")


class TestTheSnapBack:
    def _f(self):
        m, p = _markers_and_pieces()
        fake = Fake(m, p)
        _bind(fake, "_leave_reshape_review", "_marker_rows_from_refine")
        return fake, m

    def test_the_refine_roi_goes_back_to_the_markers(self):
        fake, markers = self._f()
        fake._leave_reshape_review("because")
        assert np.array_equal(
            fake._refine_layer.data[fake._split_roi], markers), (
            "the object must show its MARKERS again, not the cut")

    def test_the_result_is_nulled(self):
        fake, _ = self._f()
        fake._leave_reshape_review("because")
        assert fake._reshape_result is None
        assert fake._reshape_final_ids == {}
        assert fake._reshape_flags == {}
        assert fake._reshape_arbiter_merged is False
        assert fake._reshape_override is False

    def test_the_ids_go_back_to_the_painted_set(self):
        """`_split_seed_ids` held PIECE ids while the preview was up."""
        fake, _ = self._f()
        fake._leave_reshape_review("because")
        assert fake._split_seed_ids == [101, 102]

    def test_previewed_is_cleared(self):
        """A-H2: `_detect_seeds` early-returns on it, so a finder that
        ran before this was cleared would be a silent no-op."""
        fake, _ = self._f()
        fake._leave_reshape_review("because")
        assert fake._split_previewed is False

    def test_the_panel_is_told_review_is_over(self):
        fake, _ = self._f()
        fake._leave_reshape_review("because")
        assert any(c.get("review_active") is False for c in fake._adv.calls)
        assert any(c.get("review_active") is False for c in fake.ui_calls)

    def test_accept_is_disabled_and_the_reason_is_shown(self):
        fake, _ = self._f()
        fake._leave_reshape_review("Markers changed - press Preview.")
        acc = [c for c in fake.ui_calls if "accept_enabled" in c]
        assert acc and acc[-1]["accept_enabled"] is False
        assert any("Markers changed" in str(c.get("status_text", ""))
                   for c in fake.ui_calls)

    def test_the_subviewers_are_refreshed(self):
        fake, _ = self._f()
        fake._leave_reshape_review("because")
        assert fake.subviewer_refreshes == 1
        assert fake._refine_layer.refreshed == 1

    def test_it_is_a_no_op_when_there_is_nothing_to_leave(self):
        fake, _ = self._f()
        fake._split_previewed = False
        fake._reshape_result = None
        before = fake._refine_layer.data.copy()
        fake._leave_reshape_review("because")
        assert np.array_equal(fake._refine_layer.data, before)
        assert fake.ui_calls == []

    def test_it_is_a_no_op_with_no_session(self):
        fake, _ = self._f()
        fake._split_running = False
        fake._leave_reshape_review("because")
        assert fake.ui_calls == []

    def test_it_is_idempotent(self):
        fake, markers = self._f()
        fake._leave_reshape_review("one")
        n = len(fake.ui_calls)
        fake._leave_reshape_review("two")
        assert len(fake.ui_calls) == n
        assert np.array_equal(
            fake._refine_layer.data[fake._split_roi], markers)


class TestTheReconcilerIsAddsOnly:
    def _f(self, painted):
        m, p = _markers_and_pieces()
        fake = Fake(m, p)
        fake._split_previewed = False
        fake._refine_layer = FakeLayer(painted.copy())
        _bind(fake, "_marker_rows_from_refine")
        return fake

    def test_a_brush_painted_id_joins_the_list(self):
        painted = np.zeros((4, 4, 4), np.int32)
        painted[0, 0, 0] = 101
        painted[1, 1, 1] = 102
        painted[2, 2, 2] = 103       # painted with the napari brush
        fake = self._f(painted)
        assert fake._marker_rows_from_refine() == [101, 102, 103]
        assert fake._split_painted_marker_ids == [101, 102, 103]

    def test_the_target_is_never_taken_for_a_marker(self):
        painted = np.zeros((4, 4, 4), np.int32)
        painted[0, 0, 0] = 101
        painted[1:, :, :] = 7        # the object's own voxels
        fake = self._f(painted)
        fake._split_label_id = 7
        assert 7 not in fake._marker_rows_from_refine()

    def test_ids_BELOW_the_known_maximum_are_not_invented(self):
        """Adds-only: a piece id, or the target, would otherwise become
        a marker the moment it happened to be in the ROI."""
        painted = np.zeros((4, 4, 4), np.int32)
        painted[0, 0, 0] = 101
        painted[1, 1, 1] = 50        # below the floor
        fake = self._f(painted)
        assert fake._marker_rows_from_refine() == [101, 102]

    def test_it_refuses_to_run_during_a_preview(self):
        """A-M7: the ROI holds PIECES then, not markers."""
        painted = np.full((4, 4, 4), 900, np.int32)
        fake = self._f(painted)
        fake._split_previewed = True
        assert fake._marker_rows_from_refine() == [101, 102]
        assert fake._split_painted_marker_ids == [101, 102]


class TestUnticksReachTheSolver:
    """The belt-and-braces behind parking.

    Since an unticked marker is lifted off the Refine layer before
    Preview is ever pressed, there is normally nothing left here to
    zero.  `_apply_marker_unticks` stays as the guard for a park that
    was refused, and `test_reshape_marker_park.py` covers the parked
    cases -- including that a parked untick keeps the identity return.
    """

    def _f(self, checked):
        m, _p = _markers_and_pieces()
        fake = Fake(m, m)
        fake._split_previewed = False
        fake._adv = FakeAdv(checked=checked)
        _bind(fake, "_apply_marker_unticks", "_checked_marker_ids")
        return fake

    def test_nothing_unticked_returns_the_SAME_array(self):
        """The legacy lock: with every marker ticked the array handed to
        `seeded_watershed_split` is bit-identical -- the same object."""
        fake = self._f([101, 102])
        roi = np.zeros((2, 2, 2), np.int32)
        assert fake._apply_marker_unticks(roi) is roi

    def test_no_list_at_all_returns_the_SAME_array(self):
        fake = self._f(None)
        roi = np.zeros((2, 2, 2), np.int32)
        assert fake._apply_marker_unticks(roi) is roi

    def test_an_unticked_marker_is_zeroed_out_of_a_COPY(self):
        fake = self._f([101])
        roi = np.zeros((4, 4, 4), np.int32)
        roi[0, 0, 0] = 101
        roi[1, 1, 1] = 102
        out = fake._apply_marker_unticks(roi)
        assert out is not roi, "the layer itself must not be edited"
        assert out[0, 0, 0] == 101
        assert out[1, 1, 1] == 0
        assert roi[1, 1, 1] == 102, (
            "this filter never edits the layer it is handed -- lifting "
            "the marker off is parking's job, not this one's")


class TestEveryS3ChangeLeavesReview:
    """As source assertions: each entry point calls the one exit."""

    @pytest.mark.parametrize("method", [
        "_next_seed", "_undo_last_seed", "_refine_seeds",
        "_find_reshape_markers", "_detect_seeds",
        "_on_reshape_markers_invalidated",
        "_on_reshape_cut_knob_changed", "_on_reshape_marker_check_changed",
    ])
    def test_it_calls_leave_reshape_review(self, method):
        import inspect

        src = inspect.getsource(getattr(_K(), method))
        assert "_leave_reshape_review(" in src, (
            "{} changes S3 and must put the preview away".format(method))

    def test_a_piece_tick_does_NOT(self):
        import inspect

        src = inspect.getsource(_K()._on_reshape_pieces_changed)
        assert "_leave_reshape_review(" not in src

    def test_the_guided_preview_takes_the_snapshot(self):
        """The guided path had none, so there was nothing to snap
        back to -- only the legacy path took one."""
        import inspect

        src = inspect.getsource(_K()._preview_reshape)
        assert "_refine_pre_preview" in src

    def test_apply_reshape_result_stops_overwriting_the_markers(self):
        import inspect

        src = inspect.getsource(_K()._apply_reshape_result)
        assert "self._split_seed_ids = list(final_ids)" not in src

    def test_review_keeps_the_marker_buttons_live(self):
        import inspect

        for m in ("_preview_split", "_apply_reshape_result"):
            src = inspect.getsource(getattr(_K(), m))
            assert "next_seed_enabled=True" in src, m

    def test_the_legacy_preview_records_the_PIECES_it_made(self):
        """Accept reports `_split_seed_ids` to the controller.  Before
        the ticks existed every painted marker got a piece, so the two
        lists were the same; unticking one breaks that, and Accept would
        name a label that is not in the volume."""
        import inspect

        src = inspect.getsource(_K()._preview_split)
        assert "self._split_seed_ids = list(new_ids)" in src

    def test_and_the_snap_back_puts_the_MARKERS_back_in_it(self):
        """...which is safe only because leaving review restores it."""
        import inspect

        src = inspect.getsource(_K()._leave_reshape_review)
        assert "self._split_seed_ids = list(self._split_painted_marker_ids)"             in src


# --------------------------------------------------------------------------- #
# "Preview again" must not re-snapshot the PIECES  (Brick U3 fix, 2026-09-21)
#
# Found while building the wall picture and recorded in the UX round's
# Appendix E.  `_preview_reshape` and `_preview_split` both took
# `_refine_pre_preview` off the LIVE Refine layer, unconditionally.  In
# review that layer holds PIECES, not markers -- so "Preview again"
# pressed straight out of review overwrote the marker snapshot with the
# previous cut, and the next marker tick then "put the markers back" by
# painting that cut onto the object, with no rows behind it and
# `_marker_rows_from_refine` (whose own rule is that it never runs
# during review) ready to read it as fresh hand-painting.
# --------------------------------------------------------------------------- #
def _snapshot_block(fake):
    """The guarded write, verbatim from both `_preview_*` methods.

    Neither method can be driven past its worker launch on a Qt-free
    fake, so the BLOCK is exercised here and
    `test_both_paths_gate_the_snapshot` holds the production copies to
    the same shape.
    """
    roi = fake._split_roi
    if (fake._refine_layer is not None and roi is not None
            and not fake._review_is_on_screen()):
        fake._refine_pre_preview = fake._refine_layer.data[roi].copy()
        return True
    return False


class TestPreviewAgainKeepsTheMarkerSnapshot:
    def test_the_predicate_knows_both_paths(self):
        """`_split_previewed` is the legacy path's flag and
        `_reshape_result` the engine's; a preview on screen sets one or
        the other, so the honest predicate is the OR."""
        markers, pieces = _markers_and_pieces()
        fake = _bind(Fake(markers, pieces), "_review_is_on_screen")

        fake._split_previewed, fake._reshape_result = True, None
        assert fake._review_is_on_screen() is True
        fake._split_previewed, fake._reshape_result = False, object()
        assert fake._review_is_on_screen() is True
        fake._split_previewed, fake._reshape_result = False, None
        assert fake._review_is_on_screen() is False

    def test_it_is_NOT_retaken_while_a_preview_is_up(self):
        markers, pieces = _markers_and_pieces()
        fake = _bind(Fake(markers, pieces), "_review_is_on_screen")
        assert _snapshot_block(fake) is False, "it must not write"
        np.testing.assert_array_equal(fake._refine_pre_preview, markers)
        assert not np.array_equal(fake._refine_pre_preview, pieces)

    def test_a_snap_back_after_preview_again_shows_MARKERS(self):
        """The user-visible half, end to end: a preview is on screen,
        "Preview again" runs its snapshot block, then a marker tick
        leaves review -- and the object shows the MARKERS."""
        markers, pieces = _markers_and_pieces()
        fake = _bind(Fake(markers, pieces), "_review_is_on_screen",
                     "_leave_reshape_review", "_marker_rows_from_refine")
        _snapshot_block(fake)
        fake._leave_reshape_review("Markers changed - press Preview again.")
        roi_now = fake._refine_layer.data[fake._split_roi]
        np.testing.assert_array_equal(roi_now, markers)
        assert not np.array_equal(roi_now, pieces), (
            "the snap-back must not repaint the previous cut")

    def test_a_FRESH_preview_does_take_one(self):
        """The other half: with no preview on screen the layer holds the
        markers and the snapshot has to follow them -- otherwise a
        second object would snap back to the first one's."""
        markers, pieces = _markers_and_pieces()
        fake = _bind(Fake(markers, pieces), "_review_is_on_screen")
        fake._split_previewed = False
        fake._reshape_result = None
        fake._refine_layer.data[fake._split_roi] = markers
        fake._refine_pre_preview = None
        assert _snapshot_block(fake) is True
        np.testing.assert_array_equal(fake._refine_pre_preview, markers)

    def test_the_reconciler_does_not_run_during_review_either(self):
        """`_marker_rows_from_refine` is adds-only and its docstring
        says it must never see a preview; it rode along with the
        snapshot, so it is gated by the same predicate."""
        import inspect

        for name in ("_preview_reshape", "_preview_split"):
            src = inspect.getsource(getattr(_K(), name))
            at = src.index("_marker_rows_from_refine()")
            assert "_review_is_on_screen()" in src[:at], (
                "{} reconciles the marker list during review".format(name))

    @pytest.mark.parametrize("method", ("_preview_reshape", "_preview_split"))
    def test_both_paths_gate_the_snapshot(self, method):
        """The production copies of the block above: NEITHER may write
        the snapshot without asking whether a preview is on screen."""
        import inspect
        import re

        src = inspect.getsource(getattr(_K(), method))
        writes = [m.start() for m in
                  re.finditer(r"_refine_pre_preview = ", src)]
        assert writes, "{} no longer takes a snapshot at all".format(method)
        for at in writes:
            assert "_review_is_on_screen()" in src[:at], (
                "{} writes _refine_pre_preview unconditionally".format(
                    method))

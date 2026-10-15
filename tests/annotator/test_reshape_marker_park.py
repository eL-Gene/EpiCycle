"""Brick U2b -- an UNTICKED marker comes OFF the object.

Owner decision, 2026-09-20, on the live panel.  Until this brick an
unticked marker stayed painted on the Refine layer and the grey line
under the list said so: *"Unticked markers stay drawn - they are just
not used for the cut."*  The owner found that misleading -- the marker
you have just said is NOT an object is still a coloured blob sitting on
the object, in the 3D view and in all three side views, indistinguishable
from the ones that ARE going to cut it.

The new rule: unticking LIFTS the marker off the object, re-ticking puts
it back exactly.  It applies to found (materialised) markers and to
hand-painted ones, because since U1b they are the same voxels on the
same layer.

What must hold, and each one is a bug in the version without it:

* the voxels really leave the Refine layer, and the side views with it;
* a re-tick restores them BYTE-IDENTICALLY -- a marker is a shape, not a
  dot, and "roughly back" is a silent edit of the thing being cut;
* the snap-back never RESURRECTS a parked marker (`_refine_pre_preview`
  is taken off the live layer, so it is taken with the parked state
  already applied);
* nothing that reads "the markers on the layer" counts a parked one;
* All / None park or unpark the whole list in ONE pass with ONE
  subviewer refresh, not one per row;
* a re-Find, an Undo and the end of the session take the stash with
  them, so no voxels can be restored onto an object they no longer
  belong to.

Qt-free fakes for the orchestrator half, the offscreen panel for the
panel half -- the same split every Reshape suite uses.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")


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
    """Only the two tick questions the orchestrator asks."""

    def __init__(self, checked=None, unticked=None):
        self._checked = checked
        self._unticked = unticked
        self.calls = []

    def get_checked_marker_ids(self):
        return self._checked

    def get_unticked_marker_ids(self):
        return self._unticked

    def set_accept_reason(self, reason=""):
        pass

    def set_footer_reason(self, reason=""):
        pass

    def update_split_ui_state(self, **kw):
        self.calls.append(kw)

    def get_split_params(self):
        return {"seeds_from": 0}


class FakeSync:
    def __init__(self, shared):
        self.shared_labels = shared


class Fake:
    """A session with two markers painted on the Refine layer."""

    def __init__(self, roi=None, baseline=None, target=7):
        if roi is None:
            roi, baseline = _roi_with_two_markers()
        shape = roi.shape
        self._split_roi = (slice(0, shape[0]), slice(0, shape[1]),
                           slice(0, shape[2]))
        self._refine_layer = FakeLayer(roi.copy())
        # The ROI crop of the object as it was BEFORE any marker was
        # painted.  Parking needs it to know what a marker is sitting
        # ON -- the object, or the background beside it (H1).
        self._refine_baseline = (
            baseline.copy() if baseline is not None else None)
        self._refine_pre_preview = None
        self._split_label_id = target
        self._split_running = True
        self._split_previewed = False
        self._split_painted_marker_ids = [101, 102]
        self._split_seed_ids = [101, 102]
        self._materialised_ids = []
        self._parked_marker_voxels = {}
        self._reshape_result = None
        self._reshape_found_seeds = None
        self._active_labels_name = "curation_3d_x"
        self.sync_manager = FakeSync({"curation_3d_x": np.zeros((4, 4, 4),
                                                                np.int32)})
        self._adv = FakeAdv()
        self.ui_calls = []
        self.subviewer_refreshes = 0
        self.viewer = None

    # -- the seams the methods under test call ------------------------
    def _get_advanced_features_widget(self):
        return self._adv

    def _update_split_ui(self, **kw):
        self.ui_calls.append(kw)

    def _refresh_refine_in_subviewers(self):
        self.subviewer_refreshes += 1

    def _refresh_merge_button_state(self):
        pass

    def _refresh_label_action_buttons_state(self):
        pass


def _roi_with_two_markers():
    """``(roi, baseline)`` -- a solid object with a background rim.

    The rim matters: a marker sits ON the object, so parking has to put
    the OBJECT back under it, not a hole (H1).  A fixture that was
    target-everywhere could not tell the two apart.
    """
    baseline = np.zeros((6, 6, 6), np.int32)
    baseline[1:5, 1:5, 1:5] = 7
    roi = baseline.copy()
    roi[1:3, 1:3, 1:3] = 101
    roi[3:5, 3:5, 3:5] = 102
    return roi, baseline


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


def _parking(fake):
    return _bind(
        fake, "_sync_marker_parking", "_park_marker", "_unpark_marker",
        "_parked_marker_ids", "_unpark_all_markers",
        "_unticked_marker_ids", "_checked_marker_ids",
    )


# --------------------------------------------------------------------------- #
class TestUntickLiftsTheMarkerOff:
    def test_the_voxels_leave_the_refine_layer(self):
        fake = _parking(Fake())
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        before = int((fake._refine_layer.data == 101).sum())
        assert before == 8
        fake._sync_marker_parking()
        assert int((fake._refine_layer.data == 101).sum()) == 0
        assert int((fake._refine_layer.data == 102).sum()) == 8, (
            "the TICKED marker must not move")

    def test_it_leaves_NO_HOLE_in_the_object(self):
        """H1.  The first build wrote 0, and a marker sits ON the object
        -- so unticking punched a background-coloured bite out of the
        thing being reshaped, in 3D and in all three side views.  The
        owner asked for the marker to be hidden, not for the object to
        be damaged.  The voxels go back to whatever the BASELINE had
        there: the object where it was the object, background where the
        marker overhung the edge."""
        fake = _parking(Fake())
        zeros_before = int((fake._refine_layer.data == 0).sum())
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        assert int((fake._refine_layer.data == 0).sum()) == zeros_before, (
            "parking must not create a single new background voxel")

    def test_the_marker_is_replaced_by_the_object_underneath_it(self):
        fake = _parking(Fake())
        before = fake._refine_layer.data.copy()
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        want = before.copy()
        want[want == 101] = 0
        want[(before == 101) & (fake._refine_baseline == 7)] = 7
        assert np.array_equal(fake._refine_layer.data, want)

    def test_a_marker_overhanging_the_edge_leaves_background(self):
        """The baseline decides, voxel by voxel -- not "always target"."""
        roi, base = _roi_with_two_markers()
        roi[0, 0, 0] = 101          # outside the object, on the rim
        fake = _parking(Fake(roi=roi, baseline=base))
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        assert fake._refine_layer.data[0, 0, 0] == 0, (
            "there was no object there to put back")
        assert fake._refine_layer.data[1, 1, 1] == 7

    def test_the_object_survives_a_full_round_trip(self):
        fake = _parking(Fake())
        pristine = fake._refine_layer.data.copy()
        fake._adv._unticked = [101, 102]
        fake._sync_marker_parking()
        assert int((fake._refine_layer.data == 7).sum()) == int(
            (fake._refine_baseline == 7).sum()), (
            "with every marker lifted, the object is exactly the baseline")
        fake._adv._unticked = []
        fake._adv._checked = [101, 102]
        fake._sync_marker_parking()
        assert np.array_equal(fake._refine_layer.data, pristine)

    def test_the_marker_stays_in_the_bookkeeping(self):
        """A parked marker is still a marker: it keeps its row."""
        fake = _parking(Fake())
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        assert fake._split_painted_marker_ids == [101, 102]
        assert fake._parked_marker_ids() == [101]

    def test_the_side_views_are_refreshed(self):
        fake = _parking(Fake())
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        assert fake.subviewer_refreshes == 1

    def test_nothing_happens_when_nothing_is_unticked(self):
        fake = _parking(Fake())
        fake._adv._unticked = []
        fake._adv._checked = [101, 102]
        before = fake._refine_layer.data.copy()
        fake._sync_marker_parking()
        assert np.array_equal(fake._refine_layer.data, before)
        assert fake.subviewer_refreshes == 0, "no change, no refresh"

    def test_a_preview_on_screen_refuses_to_park(self):
        """During review the ROI holds PIECES; parking would stash one."""
        fake = _parking(Fake())
        fake._split_previewed = True
        fake._adv._unticked = [101]
        before = fake._refine_layer.data.copy()
        fake._sync_marker_parking()
        assert np.array_equal(fake._refine_layer.data, before)
        assert fake._parked_marker_ids() == []


class TestReTickRestoresItExactly:
    def test_byte_identical(self):
        fake = _parking(Fake())
        pristine = fake._refine_layer.data.copy()
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        assert not np.array_equal(fake._refine_layer.data, pristine)
        fake._adv._unticked = []
        fake._adv._checked = [101, 102]
        fake._sync_marker_parking()
        assert np.array_equal(fake._refine_layer.data, pristine), (
            "a marker is a SHAPE -- 'roughly back' is a silent edit")

    def test_the_stash_is_consumed(self):
        fake = _parking(Fake())
        fake._adv._unticked = [101]
        fake._sync_marker_parking()
        fake._adv._unticked = []
        fake._sync_marker_parking()
        assert fake._parked_marker_ids() == []

    def test_it_never_overwrites_another_marker(self):
        """The safe write: 0, the TARGET, or this id -- never another
        marker.  The target joined the set with H1, because that is
        what parking now leaves behind."""
        fake = _parking(Fake())
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        # somebody paints marker 102 over one voxel of where 101 lay
        fake._refine_layer.data[1, 1, 1] = 102
        fake._adv._unticked = []
        fake._adv._checked = [101, 102]
        fake._sync_marker_parking()
        assert fake._refine_layer.data[1, 1, 1] == 102, (
            "the re-tick must not steal a voxel from another marker")
        assert fake._refine_layer.data[2, 2, 2] == 101, (
            "...and it must take back the ones that are still the object")

    def test_one_pass_parks_before_it_unparks(self):
        """When a single pass does both, the marker that is LEAVING has
        to vacate its voxels before the one that is arriving asks for
        them, or the safe write drops the overlap."""
        fake = _parking(Fake())
        # 102 is parked, and 101 is then painted over where it lay
        fake._adv._unticked = [102]
        fake._adv._checked = [101]
        fake._sync_marker_parking()
        roi = fake._refine_layer.data[fake._split_roi]
        roi[3:5, 3:5, 3:5] = 101
        # now swap them in ONE pass
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        parked, unparked = fake._sync_marker_parking()
        assert parked == [101] and unparked == [102]
        assert int((fake._refine_layer.data == 102).sum()) == 8, (
            "102 must get its voxels back because 101 left first")
        assert int((fake._refine_layer.data == 101).sum()) == 0

    def test_a_round_trip_survives_many_times(self):
        fake = _parking(Fake())
        pristine = fake._refine_layer.data.copy()
        for _ in range(3):
            fake._adv._unticked = [101, 102]
            fake._sync_marker_parking()
            fake._adv._unticked = []
            fake._sync_marker_parking()
        assert np.array_equal(fake._refine_layer.data, pristine)


class TestAllAndNoneAreOnePass:
    def test_none_parks_every_marker_with_one_refresh(self):
        fake = _parking(Fake())
        fake._adv._unticked = [101, 102]
        fake._adv._checked = []
        fake._sync_marker_parking()
        assert sorted(fake._parked_marker_ids()) == [101, 102]
        assert fake.subviewer_refreshes == 1, (
            "one refresh for the whole list, not one per row")

    def test_all_unparks_every_marker_with_one_refresh(self):
        fake = _parking(Fake())
        pristine = fake._refine_layer.data.copy()
        fake._adv._unticked = [101, 102]
        fake._sync_marker_parking()
        fake.subviewer_refreshes = 0
        fake._adv._unticked = []
        fake._adv._checked = [101, 102]
        fake._sync_marker_parking()
        assert fake.subviewer_refreshes == 1
        assert np.array_equal(fake._refine_layer.data, pristine)


class TestTheSnapBackDoesNotResurrectIt:
    def test_the_snapshot_is_taken_with_the_parked_state_applied(self):
        """`_refine_pre_preview` is read off the LIVE layer, so a parked
        marker is absent from it and a snap-back cannot bring it back."""
        fake = _parking(Fake())
        _bind(fake, "_leave_reshape_review", "_marker_rows_from_refine")
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        # ...this is what `_preview_reshape` does, in its order
        fake._refine_pre_preview = fake._refine_layer.data[
            fake._split_roi].copy()
        assert int((fake._refine_pre_preview == 101).sum()) == 0, (
            "the snapshot must hold the PARKED state")
        # the preview writes pieces, then something leaves review
        fake._refine_layer.data[fake._split_roi] = 999
        fake._split_previewed = True
        fake._reshape_result = object()
        fake._leave_reshape_review("Markers changed - press Preview again.")
        assert int((fake._refine_layer.data == 101).sum()) == 0
        assert int((fake._refine_layer.data == 102).sum()) == 8

    def test_a_re_tick_in_review_leaves_review_then_unparks(self):
        """The order in `_on_reshape_marker_check_changed`: leave review
        first (it restores the snapshot), THEN apply the tick."""
        fake = _parking(Fake())
        _bind(fake, "_leave_reshape_review", "_marker_rows_from_refine",
              "_on_reshape_marker_check_changed")
        pristine = fake._refine_layer.data.copy()
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        fake._refine_pre_preview = fake._refine_layer.data[
            fake._split_roi].copy()
        fake._refine_layer.data[fake._split_roi] = 999
        fake._split_previewed = True
        fake._reshape_result = object()
        # the user re-ticks 101 while the cut is on screen
        fake._adv._unticked = []
        fake._adv._checked = [101, 102]
        fake._on_reshape_marker_check_changed()
        assert fake._split_previewed is False
        assert np.array_equal(fake._refine_layer.data, pristine), (
            "the snap-back puts the markers back and the re-tick puts "
            "the parked one back on top of them")


class TestTheStashGoesWithTheMarkers:
    def test_a_re_find_clears_the_stash_for_the_dropped_generation(self):
        fake = _parking(Fake())
        _bind(fake, "_drop_materialised_markers")
        fake._materialised_ids = [101]
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        assert fake._parked_marker_ids() == [101]
        fake._drop_materialised_markers(reason="before this one")
        assert fake._parked_marker_ids() == [], (
            "a dropped generation's voxels must not be restorable")
        assert 101 not in fake._split_painted_marker_ids

    def test_undo_drops_a_parked_painted_id_from_the_stash(self):
        fake = _parking(Fake())
        _bind(fake, "_undo_last_seed", "_leave_reshape_review",
              "_marker_rows_from_refine", "_marker_rows_for_panel")
        fake.viewer = _FakeViewer()
        fake._reshape_found_marker_ids = []
        fake._reshape_found_marker_stats = {}
        fake._adv._unticked = [102]
        fake._adv._checked = [101]
        fake._sync_marker_parking()
        assert fake._parked_marker_ids() == [102]
        fake._undo_last_seed()
        assert fake._split_painted_marker_ids == [101]
        assert fake._parked_marker_ids() == [], (
            "Undo removed the marker; its voxels may never come back")

    def test_the_session_reset_empties_the_stash(self):
        fake = _parking(Fake())
        _bind(fake, "_reset_split_state")
        fake._adv._unticked = [101]
        fake._sync_marker_parking()
        assert fake._parked_marker_ids() == [101]
        fake._reset_split_state()
        assert fake._parked_marker_voxels == {}

    def test_detect_drops_the_stash_because_it_resets_the_ROI(self):
        """`_apply_seed_threshold` resets the ROI to baseline and mints
        the SAME ids again (`_seed_id_base` is cached for the session),
        so a stash taken before the drag describes voxels that no longer
        exist -- and the repopulated list keeps the OFF tick by id.  It
        clears the stash and re-syncs from what the drag wrote, or the
        marker ends up drawn and unticked at once.

        MEASURED, not read (audit L2): the real method runs, and what is
        asserted is that by the time it syncs, the stale coordinates are
        gone.  A refactor that keeps both statements but swaps them
        fails here.
        """
        fake = _parking(Fake())
        _bind(fake, "_apply_seed_threshold")
        fake._detected_peaks = [{"coord": (2, 2, 2), "edt_value": 1.0}]
        fake._seed_id_base = 101
        fake._parked_marker_voxels = {101: (np.array([0]),) * 3}
        seen = {}
        fake._sync_marker_parking = lambda: seen.setdefault(
            "stash_at_sync", dict(fake._parked_marker_voxels))
        fake._apply_seed_threshold(0.0)
        assert "stash_at_sync" in seen, "it never re-synced at all"
        assert seen["stash_at_sync"] == {}, (
            "the stale coordinates must be gone BEFORE the re-sync, or "
            "it re-lifts from a layer that no longer exists")

    def test_a_stash_entry_that_is_no_longer_a_marker_is_DROPPED(self):
        """M1.  `to_unpark` used to be the whole stash, so an id that had
        left `_split_painted_marker_ids` was painted back onto the
        object -- a marker the panel cannot show and the recipe will not
        send.  Every legitimate remover pops its own ids, so one that is
        still here is a leak: drop it, do not restore it."""
        fake = _parking(Fake())
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        # ...something removed 101 from the marker set without popping
        fake._split_painted_marker_ids = [102]
        before = fake._refine_layer.data.copy()
        fake._adv._unticked = []
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        assert fake._parked_marker_ids() == [], "it must not linger either"
        assert int((fake._refine_layer.data == 101).sum()) == 0
        assert np.array_equal(fake._refine_layer.data, before)

    def test_an_emptied_list_puts_every_parked_marker_back(self):
        """No row, no tick: a marker the panel can no longer show may
        not stay lifted off, or nothing could ever put it back."""
        fake = _parking(Fake())
        pristine = fake._refine_layer.data.copy()
        fake._adv._unticked = [101]
        fake._sync_marker_parking()
        fake._adv._unticked = None       # the list was cleared
        fake._adv._checked = None
        fake._sync_marker_parking()
        assert np.array_equal(fake._refine_layer.data, pristine)


class TestNobodyCountsAParkedMarkerAsDrawn:
    def test_painted_marker_ids_on_layer_excludes_it(self):
        fake = _parking(Fake())
        _bind(fake, "_painted_marker_ids_on_layer")
        fake._adv._unticked = [101]
        fake._sync_marker_parking()
        assert fake._painted_marker_ids_on_layer() == [102]

    def test_it_excludes_it_during_review_too(self):
        """The `_split_previewed` branch answers from the LIST, which
        still holds the parked id -- so it has to subtract it."""
        fake = _parking(Fake())
        _bind(fake, "_painted_marker_ids_on_layer")
        fake._adv._unticked = [101]
        fake._sync_marker_parking()
        fake._split_previewed = True
        assert fake._painted_marker_ids_on_layer() == [102]

    def test_the_promotion_never_includes_a_parked_id(self):
        fake = _parking(Fake())
        _bind(fake, "_merged_marker_seed_layer")
        found = np.zeros((6, 6, 6), np.int32)
        found[2:4, 2:4, 2:4] = 1
        fake._reshape_found_seeds = found
        fake._adv._unticked = [101]
        fake._adv._checked = [102, 1]
        fake._sync_marker_parking()
        merged = fake._merged_marker_seed_layer()
        assert merged is not None
        assert 101 not in set(int(v) for v in np.unique(merged))
        assert 102 in set(int(v) for v in np.unique(merged))

    def test_the_recipes_wanted_filter_cannot_send_a_parked_marker(self):
        """Belt and braces: even if the tick list said otherwise, the
        voxels are not there to send."""
        fake = _parking(Fake())
        fake._adv._unticked = [101]
        fake._adv._checked = [102]
        fake._sync_marker_parking()
        roi = fake._refine_layer.data[fake._split_roi]
        wanted = [101, 102]
        seed_layer = np.where(np.isin(roi, wanted), roi, 0)
        assert int((seed_layer == 101).sum()) == 0


class TestTheLegacyUntickFilterStillHoldsItsGuarantee:
    def _f(self, checked, parked=()):
        fake = _parking(Fake())
        _bind(fake, "_apply_marker_unticks")
        fake._adv._checked = checked
        fake._parked_marker_voxels = {int(i): () for i in parked}
        return fake

    def test_nothing_unticked_returns_the_SAME_array(self):
        fake = self._f([101, 102])
        roi = np.zeros((2, 2, 2), np.int32)
        assert fake._apply_marker_unticks(roi) is roi

    def test_no_list_at_all_returns_the_SAME_array(self):
        fake = self._f(None)
        roi = np.zeros((2, 2, 2), np.int32)
        assert fake._apply_marker_unticks(roi) is roi

    def test_a_PARKED_untick_returns_the_SAME_array(self):
        """The voxels are already gone, so there is nothing to zero --
        and the legacy call stays byte-identical."""
        fake = self._f([102], parked=[101])
        roi = np.zeros((2, 2, 2), np.int32)
        assert fake._apply_marker_unticks(roi) is roi

    def test_an_unparked_untick_is_still_zeroed_out_of_a_COPY(self):
        fake = self._f([101])
        roi = np.zeros((4, 4, 4), np.int32)
        roi[0, 0, 0] = 101
        roi[1, 1, 1] = 102
        out = fake._apply_marker_unticks(roi)
        assert out is not roi
        assert out[1, 1, 1] == 0
        assert roi[1, 1, 1] == 102


class TestParkingNeverTouchesTheVolume:
    def test_only_the_refine_layer_is_written(self):
        """Discard restores the object from the split snapshot; parking
        must never have reached the shared volume for that to be true."""
        fake = _parking(Fake())
        shared = fake.sync_manager.shared_labels["curation_3d_x"]
        before = shared.copy()
        fake._adv._unticked = [101, 102]
        fake._sync_marker_parking()
        fake._adv._unticked = []
        fake._sync_marker_parking()
        assert np.array_equal(shared, before)


class TestTheMaterialiserDoesNotPruneAParkedMarker:
    def test_a_parked_hand_painted_id_keeps_its_row(self):
        """The 'an allocated id with no voxels is not a marker' prune
        would otherwise delete every parked marker on the next Find."""
        fake = _parking(Fake())
        _bind(fake, "_painted_marker_ids_on_layer", "_parked_marker_ids")
        fake._adv._unticked = [101]
        fake._sync_marker_parking()
        on_layer = fake._painted_marker_ids_on_layer()
        keepable = set(on_layer) | set(fake._parked_marker_ids())
        dropped = [i for i in fake._split_painted_marker_ids
                   if i not in keepable]
        assert dropped == []


class _FakeViewer:
    class _Layers:
        class _Sel:
            active = None

        selection = _Sel()

    layers = _Layers()


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


class TestThePanelAnswersTheOtherHalfOfTheQuestion:
    def test_unticked_ids_are_reported(self, widget):
        from qtpy.QtCore import Qt

        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_list.item(1).setCheckState(Qt.Unchecked)
        assert widget.get_unticked_marker_ids() == [2]
        assert widget.get_checked_marker_ids() == [1, 3]

    def test_no_list_answers_None_not_empty(self, widget):
        assert widget.get_unticked_marker_ids() is None

    def test_everything_ticked_answers_empty(self, widget):
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        assert widget.get_unticked_marker_ids() == []


class TestTheTickReachesTheOrchestratorOutsideReviewToo:
    def test_a_tick_fires_the_callback_with_no_preview_on_screen(
            self, widget):
        """Before U2b the callback was review-only -- a tick outside
        review had nothing to do.  Now it has to park the marker."""
        from qtpy.QtCore import Qt

        fired = []
        widget.set_split_callbacks(
            on_marker_check_changed=lambda: fired.append(1))
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_list.item(1).setCheckState(Qt.Unchecked)
        assert fired, "the orchestrator must hear every tick now"

    def test_one_tick_is_exactly_one_callback(self, widget):
        """MEASURED: it used to be three.  `_style_marker_row` calls
        `setFont` and `setForeground`, and each re-emits `itemChanged`
        -- harmless while the callback only re-rendered a count line,
        three redundant park passes once it moves voxels."""
        from qtpy.QtCore import Qt

        fired = []
        widget.set_split_callbacks(
            on_marker_check_changed=lambda: fired.append(1))
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_list.item(1).setCheckState(Qt.Unchecked)
        assert fired == [1]
        widget.split_marker_list.item(1).setCheckState(Qt.Checked)
        assert fired == [1, 1]

    def test_the_row_is_still_restyled_despite_the_block(self, widget):
        """The block is around the restyle, so the restyle had better
        still happen -- it is the only thing on the row that says the
        marker has been lifted off."""
        from qtpy.QtCore import Qt

        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        item = widget.split_marker_list.item(1)
        item.setCheckState(Qt.Unchecked)
        assert item.font().strikeOut() is True
        assert item.foreground().color().red() == 0x77

    def test_None_fires_it_once_for_the_whole_list(self, widget):
        fired = []
        widget.set_split_callbacks(
            on_marker_check_changed=lambda: fired.append(1))
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_none_btn.click()
        assert len(fired) == 1, (
            "All / None park the whole list in ONE pass")

    def test_a_PIECE_tick_still_does_not_fire_it(self, widget):
        from qtpy.QtCore import Qt

        fired = []
        widget.set_split_callbacks(
            on_marker_check_changed=lambda: fired.append(1))
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        widget.update_split_guided_ui(
            busy=False, pieces=[11, 12], piece_sizes={11: 10, 12: 20})
        widget.update_split_ui_state(review_active=True)
        widget.split_piece_list.item(0).setCheckState(Qt.Unchecked)
        assert fired == []


class TestAFindOwnsTheMarkersWhileItRuns:
    """L1.  A tick was reachable during a Find.

    `_set_marker_busy` disabled the Find button and the knobs but not
    the list -- harmless while a tick was only a filter, a race once it
    moves voxels on the very layer the Find is about to write to.  The
    solve already locks the list; now the two locks match.
    """

    def test_the_list_and_both_chips_go_dead(self, widget):
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        assert widget.split_marker_list.isEnabled() is True
        widget.update_split_ui_state(marker_busy=True)
        assert widget.split_marker_list.isEnabled() is False
        assert widget.split_marker_all_btn.isEnabled() is False
        assert widget.split_marker_none_btn.isEnabled() is False

    def test_they_all_come_back(self, widget):
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        widget.update_split_ui_state(marker_busy=True)
        widget.update_split_ui_state(marker_busy=False)
        assert widget.split_marker_list.isEnabled() is True
        assert widget.split_marker_all_btn.isEnabled() is True
        assert widget.split_marker_none_btn.isEnabled() is True


class TestTheHelpLineSaysWhatNowHappens:
    def test_the_sentence_moved_to_the_tooltip_and_the_LINE_is_gone(
            self, widget):
        """Brick U4, audit H2.  The grey line under the list cost 14 px
        in the tallest state the panel has, and since U2b the object
        answers its question by itself: untick a marker and it visibly
        leaves the object and all three side views.  The sentence stays,
        on the tooltip, where it costs nothing."""
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        assert not hasattr(widget, "split_marker_hint")
        text = widget.split_marker_list.toolTip().lower()
        assert "lifts it off" in text
        assert "stay drawn" not in text

    def test_the_list_tooltip_agrees_with_it(self, widget):
        tip = widget.split_marker_list.toolTip().lower()
        assert "stays drawn" not in tip
        assert "lift" in tip or "comes off" in tip

    def test_the_None_chip_agrees_with_it(self, widget):
        tip = widget.split_marker_none_btn.toolTip().lower()
        assert "lift" in tip or "off the object" in tip

    def test_the_All_chip_agrees_with_it_too(self, widget):
        """M2: it still said only "Tick every marker", which is now the
        smaller half of what it does."""
        tip = widget.split_marker_all_btn.toolTip().lower()
        assert "back on the object" in tip

    def test_nothing_promises_the_shape_comes_back_EXACTLY(self, widget):
        """M2.  It comes back where it was, but another marker painted
        over it in the meantime keeps the overlap -- that is the safe
        write, and a promise of "exactly" would be false in the one case
        the user could actually create."""
        tip = widget.split_marker_list.toolTip().lower()
        assert "exactly" not in tip
        assert "keeps the overlap" in tip

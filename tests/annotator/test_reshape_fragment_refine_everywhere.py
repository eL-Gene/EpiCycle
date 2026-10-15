"""Brick U4 -- "Refine Fragments..." after EVERY successful Preview.

Owner decision, 2026-09-20, on the live panel.  The hand-off that writes
the pieces into the volume and lets you smooth / dilate / grow each one
on its own was reachable only after a LEGACY (hand-painted, shape-only)
Preview: `_preview_split`'s tail was the single site that ever enabled
it, and `_apply_reshape_result` pushed `refine_further_enabled=False`
after every engine Preview.  Pick a marker source or a wall kind -- the
whole point of the Reshape tab -- and the button vanished.

Three blockers stood behind it, and each is a data-integrity bug rather
than a missing wire:

* **B1** an UNTICKED piece is drawn with the TARGET's id, and fragment
  entry treated every non-new id as a leftover and ZEROED it -- so
  unticking a piece and pressing Refine Fragments deleted that tissue;
* **B2** the fragment Accept fired the SPLIT callback, so a guided cut
  taken through fragment refine lost the `reshape_flag` sidecar, the
  `superseded=False` an unticked piece earns, the piece stats and the
  guide/preset provenance;
* **B3** fragment mode was not quiescent -- `_leave_reshape_review` was
  reachable and nulls `_reshape_result`, i.e. one marker tick silently
  destroyed the commit record, on a Refine layer that is already gone.

Qt-free fakes for the orchestrator half, the offscreen panel for the
panel half -- the same split every Reshape suite uses.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import ReshapeSpec, reshape_region

from .reshape_fixtures import two_columns


def _K():
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    return Curation3DAnnotator


def _M():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeLayer:
    def __init__(self, data, name="layer"):
        self.data = data
        self.name = name
        self.visible = True
        self.editable = True
        self.mode = "pan_zoom"
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1


class FakeSync:
    def __init__(self, shared):
        self.shared_labels = shared


class FakeAdv:
    """Just the surface the fragment + commit paths read."""

    def __init__(self, checked=None):
        M = _M()
        self._params = dict(
            seeds_from=M.SEEDS_FROM_PAINTED, walls_from=M.WALLS_FROM_SHAPE,
            guided=False, wall_channel="", solver="watershed",
            target_count=None,
        )
        self._checked = checked
        self.ui_calls = []
        self.guided_calls = []
        self.cleared = 0
        self.accept_reasons = []
        self.footer_reasons = []

    # -- what the orchestrator reads ------------------------------------ #
    def get_split_params(self):
        return dict(self._params)

    def get_checked_split_pieces(self):
        return list(self._checked) if self._checked is not None else []

    def set_checked_split_pieces(self, ids):
        self._checked = list(ids)

    def update_split_guided_ui(self, **kw):
        self.guided_calls.append(kw)
        # The real panel FILLS the piece list here, every row ticked --
        # except the slivers the quality gate names, which it unticks.
        # A fake that forgot this could not see audit L1.
        pieces = kw.get("pieces")
        if pieces is not None:
            off = set()
            for note in (kw.get("quality_notes") or []):
                if getattr(note, "kind", "") == "sliver":
                    off |= {int(i) for i in getattr(note, "piece_ids", ())}
            self._checked = [int(p) for p in pieces if int(p) not in off]

    def update_split_ui_state(self, **kw):
        self.ui_calls.append(kw)

    def clear_split_guided_ui(self):
        self.cleared += 1

    def set_accept_reason(self, reason=""):
        self.accept_reasons.append(reason)

    def set_footer_reason(self, reason=""):
        self.footer_reasons.append(reason)

    # -- the marker list, for the parking guards ------------------------ #
    def get_checked_marker_ids(self):
        return None

    def get_unticked_marker_ids(self):
        return None

    # -- the last value pushed for a kwarg ------------------------------ #
    def last(self, key, default=None):
        for call in reversed(self.ui_calls):
            if key in call:
                return call[key]
        return default


class FakeAnnotator:
    """Only the attributes the fragment / commit methods actually touch."""

    def __init__(self, volume, roi, target_id, adv=None, refine=None,
                 snapshot=None):
        self.viewer = types.SimpleNamespace(layers=[])
        self._active_labels_name = "curation_3d_x"
        self.sync_manager = FakeSync({"curation_3d_x": volume})
        self._split_roi = roi
        self._split_label_id = target_id
        self._split_snapshot = (
            volume[roi].copy() if snapshot is None else snapshot
        )
        self._split_running = True
        self._split_previewed = False
        self._split_seed_ids = []
        self._split_painted_marker_ids = []
        self._split_isolate_was_on = False
        self._split_main_visible = True
        self._split_fragment_refine = False
        self._split_fragment_ids = []
        self._fragment_refine_roi = None
        self._refine_pre_preview = None
        self._refine_layer = refine
        self._refine_baseline = None
        self._smooth_running = False
        self._load_locked = False
        self._navigator_widget = None
        # --- the guided session ------------------------------------- #
        self._reshape_active = True
        self._reshape_running = False
        self._seed_find_running = False
        self._reshape_result = None
        self._reshape_final_ids = {}
        self._reshape_flags = {}
        self._reshape_override = False
        self._reshape_arbiter_merged = False
        self._reshape_guide_run = ""
        self._reshape_fragment_ticked = None
        self._reshape_fragment_written = 0
        self._reshape_found_seeds = None
        self._reshape_found_marker_ids = []
        self._reshape_found_marker_stats = {}
        self._materialised_ids = []
        self._parked_marker_voxels = {}
        self._temp_guide_layers = {}
        self._reshape_lock_fired = False
        self._adv = adv or FakeAdv()
        # --- recorders ---------------------------------------------- #
        self.lock_calls = []
        self.ui_calls = []
        self.synced = []
        self.removed = []
        self.selected = []
        self.next_id = 100
        self.subviewer_refreshes = 0
        self.split_cb_calls = []
        self.reshape_cb_calls = []
        self.isolation_calls = []

    # --- the seams the real methods call ------------------------------- #
    def _get_advanced_features_widget(self):
        return self._adv

    def _update_split_ui(self, **kw):
        self.ui_calls.append(kw)

    def _next_label_id_after_accept(self):
        return self.next_id

    def _sync_all_views_after_operation(self, roi=None, label_id=None, **kw):
        self.synced.append((roi, label_id, kw))

    def _refresh_refine_in_subviewers(self):
        self.subviewer_refreshes += 1

    def _restore_ready_to_draw_state(self):
        pass

    def _clear_draw_snapshot(self):
        pass

    def _lock_subviewers_for_preview(self):
        pass

    def _remove_edt_overlay(self):
        self.removed.append("edt")

    def _remove_refine_layer(self):
        self.removed.append("refine")
        self._refine_layer = None

    def _remove_prob_overlay(self):
        self.removed.append("prob")

    def _reset_smooth_state(self):
        self._smooth_running = False

    def _accept_smooth(self):
        self._smooth_running = False

    def _abandon_seed_find_worker(self):
        pass

    def _abandon_reshape_worker(self):
        pass

    def _set_selected_label_all(self, label_id):
        self.selected.append(label_id)

    def _on_isolation_changed(self, ids):
        self.isolation_calls.append(set(ids))

    def _is_auto_advance_enabled(self):
        return False

    def _layer_by_name(self, layers, name):
        return None

    def _refresh_merge_button_state(self):
        pass

    def _refresh_label_action_buttons_state(self):
        pass

    def _reset_split_state(self):
        K = _K()
        K._reset_split_state.__get__(self, FakeAnnotator)()

    def _on_label_split_cb(self, original_id, new_ids, **kw):
        self.split_cb_calls.append((original_id, list(new_ids), kw))

    def _on_reshape_committed_cb(self, original_id, new_ids, **kw):
        self.reshape_cb_calls.append((original_id, list(new_ids), kw))

    def _reshape_lock_cb(self, active):
        self.lock_calls.append(bool(active))

    # --- the last value pushed for a kwarg ------------------------------ #
    def last(self, key, default=None):
        for call in reversed(self.ui_calls):
            if key in call:
                return call[key]
        return default


_ALWAYS_BOUND = (
    "_fire_reshape_lock", "_add_temp_layer", "_remove_temp_layer",
    "_remove_all_temp_layers",
    "_n_reshape_fragments", "_reshape_commit_record", "_live_piece_stats",
    "_sync_marker_parking", "_park_marker", "_unpark_marker",
    "_unpark_all_markers", "_parked_marker_ids", "_unticked_marker_ids",
    "_painted_marker_ids_on_layer",
    # Brick U3: the snap-back and the three exits drop the wall
    # picture, and they do it UNGUARDED -- a getattr dance there
    # would let a future exit skip it silently.  So every fake that
    # drives one of them binds the REAL dropper; a stub would pass
    # this file by not having the wire.
    "_drop_walls_picture", "_show_walls_picture",
    "_walls_display_crop", "_remove_temp_layer",
    # Brick B1: `_walls_display_crop` asks WHICH tool has
    # a picture staged instead of reading `_reshape_result` directly, so
    # the two accessors are part of the same real wire.
    "_active_elevation", "_active_display_roi",
    # Brick U3 fix: `_preview_split` asks this before it overwrites
    # the marker snapshot.
    "_review_is_on_screen",
)


def bind(fake, *names):
    K = _K()
    for name in tuple(names) + _ALWAYS_BOUND:
        setattr(fake, name, getattr(K, name).__get__(fake, FakeAnnotator))
    return fake


# --------------------------------------------------------------------------- #
# a real two-piece engine result
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def solved():
    labels, guide, membrane, _ = two_columns(0.30)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    res = reshape_region(labels, {"membrane": membrane}, spec,
                         target_id=7, seed_layer=guide)
    assert res.status == "ok" and res.n_pieces >= 2
    return labels, res


def _guided_fake(solved, checked=None, target=7):
    """A session sitting in guided REVIEW, with the cut painted."""
    labels, res = solved
    roi = res.roi
    vol = labels.copy()
    snapshot = vol[roi].copy()          # BEFORE the hole is punched
    vol[vol == target] = 0              # the hole `_start_split` makes
    refine = FakeLayer(np.zeros_like(labels), "Refine")
    piece_ids = [int(r["id"]) for r in res.piece_stats]
    final = {p: 100 + i for i, p in enumerate(piece_ids)}
    adv = FakeAdv(checked=list(final.values()) if checked is None else checked)
    fake = FakeAnnotator(vol, roi, target, adv=adv, refine=refine,
                         snapshot=snapshot)
    fake._reshape_result = res
    fake._reshape_final_ids = dict(final)
    fake._split_previewed = True
    bind(fake, "_render_reshape_preview", "_enter_fragment_refine_mode",
         "_accept_split", "_reject_split", "_leave_reshape_review",
         "_write_reshape_result", "_on_reshape_pieces_changed",
         "_restore_reshape_target")
    fake._render_reshape_preview()
    return fake, res, final


# --------------------------------------------------------------------------- #
# 1. THE ASK: the button is offered after an ENGINE preview
# --------------------------------------------------------------------------- #
class TestTheButtonIsOfferedAfterAnEnginePreview:
    def _applied(self, solved, next_id=100):
        labels, res = solved
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        fake = FakeAnnotator(labels.copy(), res.roi, 7, refine=refine)
        fake.next_id = next_id
        bind(fake, "_apply_reshape_result", "_print_count_diag_result",
             "_render_reshape_preview", "_reshape_quality_notes",
             "_reshape_review_status")
        fake._apply_reshape_result(res)
        return fake

    def test_a_guided_preview_enables_refine_fragments(self, solved):
        """The whole brick, in one assertion."""
        fake = self._applied(solved)
        assert fake._adv is not None
        last = None
        for call in fake.ui_calls:
            if "refine_further_enabled" in call:
                last = call["refine_further_enabled"]
        assert last is True, (
            "an engine Preview must offer Refine Fragments, the same as a "
            "hand-painted one")

    def test_the_legacy_preview_tail_still_enables_it(self):
        """Driven, not read (audit L2).  A hand-painted two-seed preview
        must still offer the hand-off it has always offered."""
        shape = (6, 10, 10)
        labels = np.zeros(shape, np.int32)
        labels[1:5, 1:9, 1:9] = 7
        roi = (slice(0, 6), slice(0, 10), slice(0, 10))
        vol = labels.copy()
        vol[vol == 7] = 0
        refine = FakeLayer(labels.copy(), "Refine")
        refine.data[2, 3, 3] = 101
        refine.data[2, 6, 6] = 102
        fake = FakeAnnotator(vol, roi, 7, refine=refine,
                             snapshot=labels[roi].copy())
        fake._reshape_active = False
        fake._refine_baseline = labels[roi].copy()
        fake._split_painted_marker_ids = [101, 102]
        bind(fake, "_preview_split", "_leave_reshape_review",
             "_marker_rows_from_refine", "_apply_marker_unticks",
             "_painted_marker_ids_on_layer", "_checked_marker_ids")
        # the routing seams: this test is about the LEGACY tail
        fake._arm_guided_if_needed = lambda: False
        fake._is_guided_now = lambda: False

        fake._preview_split()

        assert fake._split_previewed is True, fake.ui_calls[-1]
        assert fake.last("refine_further_enabled") is True

    def test_the_legacy_preview_leaves_no_engine_result_behind(self):
        """Audit L3.  Accept routes on `_reshape_result`, so a stale one
        would send a WATERSHED preview down the engine's commit."""
        shape = (6, 10, 10)
        labels = np.zeros(shape, np.int32)
        labels[1:5, 1:9, 1:9] = 7
        roi = (slice(0, 6), slice(0, 10), slice(0, 10))
        vol = labels.copy()
        vol[vol == 7] = 0
        refine = FakeLayer(labels.copy(), "Refine")
        refine.data[2, 3, 3] = 101
        refine.data[2, 6, 6] = 102
        fake = FakeAnnotator(vol, roi, 7, refine=refine,
                             snapshot=labels[roi].copy())
        fake._reshape_active = False
        fake._refine_baseline = labels[roi].copy()
        fake._split_painted_marker_ids = [101, 102]
        fake._reshape_result = object()          # a leftover
        bind(fake, "_preview_split", "_leave_reshape_review",
             "_marker_rows_from_refine", "_apply_marker_unticks",
             "_painted_marker_ids_on_layer", "_checked_marker_ids")
        # the routing seams: this test is about the LEGACY tail
        fake._arm_guided_if_needed = lambda: False
        fake._is_guided_now = lambda: False

        fake._preview_split()

        assert fake._reshape_result is None

    def test_kept_whole_does_NOT_offer_it(self, solved):
        """The arbiter kept the object whole: every final id IS the
        target's, so there are no fragments -- only the object."""
        labels, res = solved
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        fake = FakeAnnotator(labels.copy(), res.roi, 7, refine=refine)
        bind(fake, "_n_reshape_fragments")
        fake._reshape_final_ids = {
            int(r["id"]): 7 for r in res.piece_stats
        }
        assert fake._n_reshape_fragments(ticked={7}) == 0

    def test_every_piece_unticked_does_NOT_offer_it(self, solved):
        fake, res, final = _guided_fake(solved, checked=[])
        fake._on_reshape_pieces_changed()
        assert fake.last("refine_further_enabled") is False
        assert fake.last("accept_enabled") is False

    def test_one_ticked_piece_DOES_offer_it(self, solved):
        """Not `n_pieces >= 2`: with one piece ticked there is exactly one
        fragment to refine, and refining it is the point."""
        fake, res, final = _guided_fake(solved)
        one = sorted(final.values())[:1]
        fake._adv.set_checked_split_pieces(one)
        fake._on_reshape_pieces_changed()
        assert fake.last("refine_further_enabled") is True

    def test_the_gate_s_unticked_slivers_are_read_LIVE(self, solved):
        """Audit L1.  `_apply_reshape_result` used to compute the button
        from ALL the final ids on the theory that "every row arrives
        ticked" -- but `update_split_guided_ui` has just run, and the
        quality gate unticks the slivers it names.  With every piece a
        sliver there is nothing to refine, and the button must be dark."""
        from packages.epicycle_annotator.reshape import QualityNote

        labels, res = solved
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        fake = FakeAnnotator(labels.copy(), res.roi, 7, refine=refine)
        bind(fake, "_apply_reshape_result", "_print_count_diag_result",
             "_render_reshape_preview", "_reshape_quality_notes",
             "_reshape_review_status")
        # the gate calls every piece a sliver
        fake._reshape_quality_notes = lambda result: [
            QualityNote("sliver", "tiny",
                        tuple(range(fake.next_id,
                                    fake.next_id + len(res.piece_stats))))
        ]

        fake._apply_reshape_result(res)

        assert fake._adv.get_checked_split_pieces() == []
        assert fake.last("refine_further_enabled") is False, (
            "the button was lit for a cut whose only pieces the gate had "
            "just thrown out")
        assert fake.last("accept_enabled") is not None

    def test_a_snap_back_hides_it_again(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._leave_reshape_review("Markers changed - press Preview again.")
        vals = [c["refine_further_enabled"] for c in fake.ui_calls
                if "refine_further_enabled" in c]
        assert vals and vals[-1] is False


# --------------------------------------------------------------------------- #
# 1b. the wall picture rides through the hand-off (U3 fix pass, M3)
# --------------------------------------------------------------------------- #
class TestTheWallPictureSurvivesFragmentEntry:
    """`docs/guides/annotator_3d_reshape.md` tells the user the wall
    picture stays up through "Refine Fragments...", so the claim is
    DRIVEN here rather than read off the method.

    It survives because fragment entry pushes `review_active=True` --
    the picture belongs to the cut that made these fragments and is
    still the honest answer to "why did it cut there" while you shape
    them.  Nothing about that is obvious from the code, which is exactly
    why the guide sentence needs a lock.
    """

    def _with_a_picture(self, solved):
        fake, res, final = _guided_fake(solved)
        volume = fake.sync_manager.shared_labels["curation_3d_x"]
        main = FakeLayer(volume, "curation_3d_x")
        fake.viewer.layers.append(main)
        fake._layer_by_name = lambda layers, name: next(
            (l for l in layers if getattr(l, "name", None) == name), None)
        bind(fake, "_on_reshape_show_walls_toggled")
        fake._on_reshape_show_walls_toggled(True)
        assert "walls" in fake._temp_guide_layers, "the fixture must show it"
        return fake

    def test_the_picture_is_still_up_afterwards(self, solved):
        fake = self._with_a_picture(solved)
        fake._enter_fragment_refine_mode()
        assert "walls" in fake._temp_guide_layers
        assert [l for l in fake.viewer.layers
                if getattr(l, "name", "") == "guide_walls"]

    def test_the_intent_is_still_held(self, solved):
        """So a later re-Preview would rebuild it rather than silently
        leaving it off."""
        fake = self._with_a_picture(solved)
        fake._enter_fragment_refine_mode()
        assert fake._walls_wanted is True

    def test_discarding_out_of_fragment_mode_DOES_take_it_away(self, solved):
        """The other half of the claim: it stays up for the hand-off and
        goes on the exit, like every other exit."""
        fake = self._with_a_picture(solved)
        fake._enter_fragment_refine_mode()
        fake._reject_split()
        assert "walls" not in fake._temp_guide_layers
        assert fake._walls_wanted is False


# --------------------------------------------------------------------------- #
# 2. B1 -- entry honours the ticks, and never deletes tissue
# --------------------------------------------------------------------------- #
class TestEntryHonoursTheTicks:
    def test_an_unticked_piece_keeps_the_PARENT_id(self, solved):
        fake, res, final = _guided_fake(solved)
        keep, drop = sorted(final.values())
        fake._adv.set_checked_split_pieces([keep])
        fake._render_reshape_preview()       # redraw with the tick off

        roi_before = fake._refine_layer.data[fake._split_roi].copy()
        fake._enter_fragment_refine_mode()

        vol_roi = fake.sync_manager.shared_labels["curation_3d_x"][
            fake._split_roi]
        # the ticked piece became its own object ...
        assert int((vol_roi == keep).sum()) == int((roi_before == keep).sum())
        assert int((vol_roi == keep).sum()) > 0
        # ... and the unticked one went BACK to the parent, not to zero.
        unticked = int((roi_before == 7).sum())
        assert unticked > 0, "the fixture must actually have an unticked piece"
        assert int((vol_roi == 7).sum()) == unticked, (
            "B1: an unticked piece is drawn with the target's id and was "
            "being ZEROED -- that deletes tissue")
        assert drop not in set(np.unique(vol_roi))

    def test_the_fragment_list_holds_only_the_TICKED_pieces(self, solved):
        fake, res, final = _guided_fake(solved)
        keep, drop = sorted(final.values())
        fake._adv.set_checked_split_pieces([keep])
        fake._render_reshape_preview()
        fake._enter_fragment_refine_mode()
        assert fake._split_fragment_ids == [keep]
        assert fake._split_fragment_refine is True

    def test_nothing_of_the_object_is_lost_that_a_normal_accept_would_keep(
            self, solved):
        """The invariant, stated as a voxel count: fragment entry writes
        exactly what `_write_reshape_result` would have written."""
        fake_a, res, final = _guided_fake(solved)
        keep, _drop = sorted(final.values())
        fake_a._adv.set_checked_split_pieces([keep])
        fake_a._render_reshape_preview()
        fake_a._enter_fragment_refine_mode()
        via_fragment = fake_a.sync_manager.shared_labels["curation_3d_x"].copy()

        fake_b, _res, _final = _guided_fake(solved)
        fake_b._adv.set_checked_split_pieces([keep])
        fake_b._render_reshape_preview()
        vol_b = fake_b.sync_manager.shared_labels["curation_3d_x"]
        fake_b._write_reshape_result(vol_b, 7)

        assert np.array_equal(via_fragment, vol_b), (
            "Refine Fragments must put the same voxels in the volume that "
            "Accept would have")

    def test_all_ticked_is_the_old_behaviour(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()
        vol_roi = fake.sync_manager.shared_labels["curation_3d_x"][
            fake._split_roi]
        assert sorted(fake._split_fragment_ids) == sorted(final.values())
        assert 7 not in set(np.unique(vol_roi))

    def test_a_neighbour_label_is_never_touched(self, solved):
        labels, res = solved
        roi = res.roi
        vol = labels.copy()
        vol[0, 0, 0] = 99                     # a neighbour inside the ROI
        snapshot = vol[roi].copy()
        vol[vol == 7] = 0
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        piece_ids = [int(r["id"]) for r in res.piece_stats]
        final = {p: 100 + i for i, p in enumerate(piece_ids)}
        fake = FakeAnnotator(vol, roi, 7,
                             adv=FakeAdv(checked=list(final.values())),
                             refine=refine, snapshot=snapshot)
        fake._reshape_result = res
        fake._reshape_final_ids = dict(final)
        fake._split_previewed = True
        bind(fake, "_render_reshape_preview", "_enter_fragment_refine_mode")
        fake._render_reshape_preview()
        fake._enter_fragment_refine_mode()
        assert fake.sync_manager.shared_labels["curation_3d_x"][0, 0, 0] == 99


# --------------------------------------------------------------------------- #
# 3. the LEGACY path is byte-identical
# --------------------------------------------------------------------------- #
class TestTheLegacyPathIsByteIdentical:
    def _legacy(self):
        """A hand-painted session: pieces 101/102 plus watershed residue."""
        shape = (6, 8, 8)
        labels = np.zeros(shape, np.int32)
        labels[1:5, 1:7, 1:7] = 7
        roi = (slice(0, 6), slice(0, 8), slice(0, 8))
        snapshot = labels[roi].copy()
        vol = labels.copy()
        vol[vol == 7] = 0
        refine_data = np.zeros(shape, np.int32)
        refine_data[1:5, 1:4, 1:7] = 101
        refine_data[1:5, 4:6, 1:7] = 102
        refine_data[1:5, 6:7, 1:7] = 7      # unassigned residue -> zeroed
        refine = FakeLayer(refine_data, "Refine")
        fake = FakeAnnotator(vol, roi, 7, refine=refine, snapshot=snapshot)
        fake._reshape_active = False
        fake._reshape_result = None          # THE legacy discriminator
        fake._split_previewed = True
        fake._split_seed_ids = [101, 102]
        bind(fake, "_enter_fragment_refine_mode", "_accept_split",
             "_reject_split")
        return fake, refine_data.copy(), snapshot

    def test_entry_writes_exactly_what_the_old_code_wrote(self):
        fake, refine_roi, snapshot = self._legacy()
        expected = fake.sync_manager.shared_labels["curation_3d_x"].copy()
        roi = fake._split_roi
        original_mask = snapshot == 7
        fragment_mask = (refine_roi > 0) & (refine_roi != 7)
        write_mask = fragment_mask & original_mask
        expected[roi][write_mask] = refine_roi[write_mask]
        expected[roi][original_mask & ~fragment_mask] = 0

        fake._enter_fragment_refine_mode()

        assert np.array_equal(
            fake.sync_manager.shared_labels["curation_3d_x"], expected), (
            "the legacy fragment write must stay byte-identical -- an "
            "orig-id voxel there is watershed residue, not an unticked piece")
        assert fake._split_fragment_ids == [101, 102]

    def test_a_legacy_accept_fires_the_SPLIT_callback_and_no_other(self):
        fake, _refine, _snap = self._legacy()
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        assert len(fake.split_cb_calls) == 1
        assert fake.reshape_cb_calls == []
        assert fake.split_cb_calls[0][1] == [101, 102]

    def test_the_keep_mask_is_reached_ONLY_with_an_engine_result(self):
        """Measured, not read (audit L2).  The SAME Refine ROI, with and
        without a result in hand: with one, the orig-id voxels are an
        unticked piece and are kept; without one they are unassigned
        watershed residue and are zeroed.  That difference IS the
        guard."""
        def _run(with_result):
            fake, refine_roi, snapshot = self._legacy()
            if with_result:
                fake._reshape_result = object()
            fake._enter_fragment_refine_mode()
            vol = fake.sync_manager.shared_labels["curation_3d_x"]
            return int((vol[fake._split_roi] == 7).sum())

        residue = 4 * 1 * 6                     # the 7-valued slab
        assert _run(False) == 0, "legacy residue must still be zeroed"
        assert _run(True) == residue, (
            "with a result those voxels are an unticked piece")


# --------------------------------------------------------------------------- #
# 4. B2 -- Accept all routes through the reshape commit
# --------------------------------------------------------------------------- #
class TestAcceptAllRoutesThroughTheReshapeCommit:
    def test_it_fires_the_reshape_callback_not_the_split_one(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        assert fake.split_cb_calls == [], (
            "B2: a guided cut must not be reported as a manual split")
        assert len(fake.reshape_cb_calls) == 1

    def test_the_callback_carries_the_whole_bookkeeping(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._reshape_guide_run = "run-a"
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        orig, new_ids, kw = fake.reshape_cb_calls[0]
        assert orig == 7
        assert sorted(new_ids) == sorted(final.values())
        for key in ("flags", "superseded", "piece_stats", "guide_run",
                    "preset", "roi_slices", "old_roi_data"):
            assert key in kw, "missing {} -- the sidecar needs it".format(key)
        assert kw["guide_run"] == "run-a"
        assert kw["superseded"] is True
        assert isinstance(kw["piece_stats"], list) and kw["piece_stats"]
        assert kw["roi_slices"] == res.roi

    def test_an_unticked_piece_makes_superseded_FALSE(self, solved):
        """The target SURVIVES -- its voxels are in the volume under its own
        id -- so its row must stay active."""
        fake, res, final = _guided_fake(solved)
        keep, _drop = sorted(final.values())
        fake._adv.set_checked_split_pieces([keep])
        fake._render_reshape_preview()
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        _orig, new_ids, kw = fake.reshape_cb_calls[0]
        assert new_ids == [keep]
        assert kw["superseded"] is False
        vol_roi = fake.sync_manager.shared_labels["curation_3d_x"][res.roi]
        assert int((vol_roi == 7).sum()) > 0, (
            "superseded=False must be TRUE of the volume, not just of the "
            "callback")

    def test_the_stats_are_RE_MEASURED_after_the_fragment_edits(self, solved):
        """Audit M1.  `_reshape_commit_record` reports the ENGINE's
        `piece_stats`, measured before the user shaped anything -- and
        fragment refine exists precisely so they can shape it.  The
        driver measured 14,930 voxels recorded for a fragment that was
        13,616 by the time Accept fired.  The legacy callback always
        recomputed from the live volume; this one must too."""
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()

        # a "smooth": shave a slab off the first fragment, the way
        # `_accept_smooth` leaves the volume
        vol = fake.sync_manager.shared_labels["curation_3d_x"]
        roi = fake._split_roi
        crop = vol[roi]
        victim = sorted(final.values())[0]
        engine_voxels = {int(r["id"]): int(r["voxels"])
                         for r in res.piece_stats}
        before = int((crop == victim).sum())
        zs = np.argwhere(crop == victim)[:, 0]
        crop[(crop == victim) & (np.arange(crop.shape[0])[:, None, None]
                                 == int(zs.min()))] = 0
        after = int((crop == victim).sum())
        assert 0 < after < before, "the fake smooth must actually shrink it"

        fake._accept_split()
        _orig, _new_ids, kw = fake.reshape_cb_calls[0]
        stats = {int(r["id"]): int(r["voxels"]) for r in kw["piece_stats"]}
        assert stats[victim] == after, (
            "the df would carry {} for a fragment that is {}".format(
                stats.get(victim), after))
        assert stats[victim] != max(engine_voxels.values()), (
            "that is the engine's pre-edit number")

    def test_the_re_measured_centroids_are_in_VOLUME_coordinates(self, solved):
        """The controller multiplies them by the downsample factor and
        stores them as canonical full-res coords, so a ROI-relative
        centroid puts every fragment in the wrong place, silently."""
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        _orig, _new_ids, kw = fake.reshape_cb_calls[0]
        vol = fake.sync_manager.shared_labels["curation_3d_x"]
        for row in kw["piece_stats"]:
            coords = np.argwhere(vol == int(row["id"]))
            expect = coords.mean(axis=0)
            got = np.asarray(row["centroid_zyx"], float)
            assert np.allclose(got, expect), (
                "{}: {} vs {}".format(row["id"], got, expect))

    def test_an_unticked_piece_is_not_re_measured_into_the_stats(self, solved):
        """It went back to the TARGET id, and the target is not a new
        cell -- the engine's own filter said so and the rebuild must
        keep saying so."""
        fake, res, final = _guided_fake(solved)
        keep, drop = sorted(final.values())
        fake._adv.set_checked_split_pieces([keep])
        fake._render_reshape_preview()
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        _orig, _new_ids, kw = fake.reshape_cb_calls[0]
        assert [int(r["id"]) for r in kw["piece_stats"]] == [keep]

    def test_the_record_is_built_before_the_state_reset(self, solved):
        """Ordered by observation, not by reading the source (audit L2).
        `_reset_split_state` nulls `_reshape_result`, so a record built
        after it would be `None` and the callback would never fire -- and
        the recorder proves the record survived with its contents."""
        fake, res, final = _guided_fake(solved)
        order = []
        _real_reset = fake._reset_split_state
        _real_record = fake._reshape_commit_record

        def _reset():
            order.append("reset")
            _real_reset()

        def _record(*a, **kw):
            order.append("record")
            assert fake._reshape_result is not None, (
                "the result is already gone")
            return _real_record(*a, **kw)

        fake._reset_split_state = _reset
        fake._reshape_commit_record = _record
        fake._enter_fragment_refine_mode()
        fake._accept_split()

        assert order == ["record", "reset"]
        assert len(fake.reshape_cb_calls) == 1

    def test_a_dividing_verdict_still_commits(self, solved):
        """`new_ids == []` is a NO-OP write whose verdict is the tool's most
        important answer -- it must still reach the controller."""
        fake, res, final = _guided_fake(solved)
        bind(fake, "_reshape_commit_record")
        fake._reshape_final_ids = {p: 7 for p in final}
        record = fake._reshape_commit_record(7, ticked={7}, written=0, t0=0.0)
        assert record["new_ids"] == []
        assert record["superseded"] is False
        assert record["flags"].get(7) == "dividing"

    def test_an_override_is_recorded(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._reshape_override = True
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        _orig, new_ids, kw = fake.reshape_cb_calls[0]
        assert set(kw["flags"].values()) == {"split-override"}

    def test_the_marker_ticks_are_cleared_on_the_fragment_commit(self, solved):
        """Bug A's gate applies here too: the markers belonged to the object
        that has just been written."""
        fake, res, final = _guided_fake(solved)
        fake._split_painted_marker_ids = [900, 901]
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        assert fake._adv.cleared >= 1
        assert fake._split_painted_marker_ids == []

    def test_the_lock_comes_off_at_the_end(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._fire_reshape_lock(True)
        fake._enter_fragment_refine_mode()
        assert fake.lock_calls == [True], (
            "entry must not release the lock -- the session is still open")
        fake._accept_split()
        assert fake.lock_calls[-1] is False


# --------------------------------------------------------------------------- #
# 5. B3 -- the session is quiescent in fragment mode
# --------------------------------------------------------------------------- #
class TestWhatS1SaysInFragmentMode:
    """Audit M2.  "Fragment refine" is the method's name, not something
    the user chose -- what they pressed was "Refine Fragments…"."""

    def test_the_entry_line_is_in_plain_words(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()
        line = str(fake.last("status_text", ""))
        assert "Refining the pieces" in line
        assert "now in the volume" in line
        assert "markers are put away" in line
        for jargon in ("Fragment refine:", "Select + Refine"):
            assert jargon not in line

    def test_the_accept_line_names_the_OBJECT(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        lines = [str(c["status_text"]) for c in fake.ui_calls
                 if "status_text" in c]
        accepted = [ln for ln in lines if "accepted" in ln.lower()]
        assert accepted, lines[-3:]
        assert accepted[-1].startswith("Pieces accepted. Object 7")
        assert "Fragment refine accepted" not in accepted[-1]


class TestFragmentModeIsQuiescent:
    def test_leave_review_is_inert(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()
        before = fake._reshape_result
        n_calls = len(fake.ui_calls)
        fake._leave_reshape_review("a marker tick that must do nothing")
        assert fake._reshape_result is before, (
            "B3: leaving review in fragment mode nulls the commit record")
        assert fake._reshape_final_ids == final
        assert len(fake.ui_calls) == n_calls

    def test_entry_refuses_while_a_solve_is_in_flight(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._reshape_running = True
        fake._enter_fragment_refine_mode()
        assert fake._split_fragment_refine is False
        assert fake._refine_layer is not None

    def test_entry_refuses_while_a_find_is_in_flight(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._seed_find_running = True
        fake._enter_fragment_refine_mode()
        assert fake._split_fragment_refine is False

    def test_a_patch_switch_out_of_fragment_mode_clears_the_panel(
            self, solved):
        """Audit H1, the orchestrator half: the teardown's ONE panel call
        is `clear_split_guided_ui`, and nothing else pushes
        `fragment_refine_active=False` on that route."""
        fake, res, final = _guided_fake(solved)
        bind(fake, "_teardown_reshape_for_switch", "_restore_reshape_target")
        fake._abandon_reshape_worker = lambda: None
        fake._abandon_seed_find_worker = lambda: None
        fake._guide_thread_cleanup = lambda: None
        fake._remove_guide_layer = lambda: None
        fake._guide_pending = None
        fake._enter_fragment_refine_mode()
        before = fake._adv.cleared

        fake._teardown_reshape_for_switch()

        assert fake._adv.cleared == before + 1
        assert fake._split_fragment_refine is False
        assert fake._reshape_fragment_ticked is None

    def test_parking_refuses_in_fragment_mode(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()
        parked, unparked = fake._sync_marker_parking()
        assert (parked, unparked) == ([], [])


# --------------------------------------------------------------------------- #
# 6. Discard from fragment mode
# --------------------------------------------------------------------------- #
class TestDiscardFromFragmentMode:
    def test_the_object_comes_back_whole(self, solved):
        fake, res, final = _guided_fake(solved)
        keep, _drop = sorted(final.values())
        fake._adv.set_checked_split_pieces([keep])
        fake._render_reshape_preview()
        snapshot = fake._split_snapshot.copy()
        fake._enter_fragment_refine_mode()
        fake._reject_split()
        vol_roi = fake.sync_manager.shared_labels["curation_3d_x"][res.roi]
        assert np.array_equal(vol_roi, snapshot), (
            "Discard from fragment refine must put the object back whole")

    def test_it_comes_back_whole_with_markers_parked(self, solved):
        fake, res, final = _guided_fake(solved)
        snapshot = fake._split_snapshot.copy()
        fake._split_painted_marker_ids = [900, 901]
        fake._parked_marker_voxels = {
            901: tuple(np.array([0], dtype=np.intp) for _ in range(3))
        }
        fake._enter_fragment_refine_mode()
        fake._reject_split()
        vol_roi = fake.sync_manager.shared_labels["curation_3d_x"][res.roi]
        assert np.array_equal(vol_roi, snapshot)
        assert fake._parked_marker_voxels == {}, (
            "the stash holds ROI coordinates of a session that has ended")

    def test_discard_fires_no_commit_callback(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._enter_fragment_refine_mode()
        fake._reject_split()
        assert fake.reshape_cb_calls == []
        assert fake.split_cb_calls == []

    def test_the_lock_comes_off(self, solved):
        fake, res, final = _guided_fake(solved)
        fake._fire_reshape_lock(True)
        fake._enter_fragment_refine_mode()
        fake._reject_split()
        assert fake.lock_calls[-1] is False


# --------------------------------------------------------------------------- #
# 7. the panel half -- the marker work is locked in fragment mode
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    assert qtpy
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


class TestThePanelLocksTheMarkerWork:
    def _into_fragment_mode(self, widget):
        widget.update_split_ui_state(
            split_active=True, seed_ids=[9, 10], review_active=True,
            reject_text="Discard",
        )
        widget.update_split_ui_state(
            fragment_refine_active=True, fragment_ids=[11, 12],
            active_fragment_id=11, accept_text="Accept all (2)",
            reject_text="Discard", refine_further_enabled=False,
        )

    def test_sources_collapse_in_fragment_mode(self, widget):
        self._into_fragment_mode(widget)
        assert widget._split_sources_locked is True
        assert widget.split_guided_summary_btn.isEnabled() is False

    def test_the_marker_list_is_dead_in_fragment_mode(self, widget):
        self._into_fragment_mode(widget)
        for attr in ("split_marker_list", "split_marker_all_btn",
                     "split_marker_none_btn", "split_next_seed_btn",
                     "split_undo_seed_btn", "split_refine_seeds_btn"):
            w = getattr(widget, attr, None)
            assert w is not None
            assert w.isEnabled() is False, "{} is live in fragment mode".format(
                attr)

    def test_leaving_fragment_mode_gives_the_work_back(self, widget):
        widget.update_split_ui_state(
            split_active=True, seed_ids=[9, 10], next_seed_enabled=True,
        )
        before = widget.split_next_seed_btn.isEnabled()
        self._into_fragment_mode(widget)
        widget.update_split_ui_state(
            split_active=False, fragment_refine_active=False,
            reject_text="Cancel",
        )
        assert widget._split_sources_locked is False
        assert widget.split_next_seed_btn.isEnabled() == before

    def test_the_secondary_still_reads_Discard_through_fragment_refine(
            self, widget):
        self._into_fragment_mode(widget)
        assert widget.split_reject_btn.text() == "Discard"

    def test_the_tooltip_still_keeps_its_two_promises(self, widget):
        tip = widget.split_refine_further_btn.toolTip().lower()
        assert "refine tab" not in tip
        assert "one at a time" in tip

    def test_a_TEARDOWN_out_of_fragment_mode_gives_the_work_back(self, widget):
        """Audit H1.  A patch switch does not push
        `fragment_refine_active=False` -- it calls `clear_split_guided_ui`,
        which used to push `review_active=False` and nothing else.  The
        flag stayed True and S3 plus the whole marker block stayed dead
        for the REST of the annotator session, on the next object and
        every object after it."""
        widget.update_split_ui_state(
            split_active=True, seed_ids=[9, 10], next_seed_enabled=True,
        )
        before = widget.split_next_seed_btn.isEnabled()
        self._into_fragment_mode(widget)
        assert widget._split_fragment_active is True

        # the patch switch: `_teardown_reshape_for_switch`'s one panel call
        widget.clear_split_guided_ui()

        assert widget._split_fragment_active is False
        assert widget._split_sources_locked is False
        assert widget.split_guided_summary_btn.isEnabled() is True
        assert widget.split_next_seed_btn.isEnabled() == before
        assert widget.split_marker_list.isEnabled() is True

    def test_the_next_session_after_a_teardown_is_live(self, widget):
        """...and the state that matters is the NEXT object's."""
        self._into_fragment_mode(widget)
        widget.clear_split_guided_ui()
        widget.update_split_ui_state(
            split_active=True, seed_ids=[4, 5], next_seed_enabled=True,
        )
        assert widget.split_next_seed_btn.isEnabled() is True
        assert widget.split_marker_list.isEnabled() is True
        assert widget.split_guided_summary_btn.isEnabled() is True


class TestUntickingEveryPieceSaysWhy:
    """Audit H3.  Accept greys and the hand-off button VANISHES, and
    neither of them said a word about why."""

    def test_no_ticked_piece_puts_the_reason_on_Accept_and_the_footer(
            self, widget):
        widget.update_split_ui_state(split_active=True)
        widget.update_split_guided_ui(
            pieces=[11, 12], piece_sizes={11: 400, 12: 380},
            caption="", arbiter_message="", split_anyway_visible=False,
        )
        widget.update_split_ui_state(review_active=True, accept_enabled=True)
        widget.set_accept_reason(
            "Nothing ticked - tick a piece to Accept it or to refine the "
            "fragments.")
        widget.set_footer_reason(
            "Nothing ticked - tick a piece to Accept it or to refine the "
            "fragments.")
        assert "nothing ticked" in widget.split_accept_btn.toolTip().lower()
        assert "nothing ticked" in widget._split_footer_reason.lower()

    def test_the_orchestrator_sets_it_at_zero_and_clears_it_above_zero(
            self, solved):
        fake, res, final = _guided_fake(solved, checked=[])
        fake._on_reshape_pieces_changed()
        assert fake._adv.accept_reasons[-1], (
            "H3: Accept greys and the button vanishes with no words")
        assert "tick a piece" in fake._adv.accept_reasons[-1].lower()
        assert fake._adv.footer_reasons[-1] == fake._adv.accept_reasons[-1]

        fake._adv.set_checked_split_pieces(sorted(final.values()))
        fake._on_reshape_pieces_changed()
        assert fake._adv.accept_reasons[-1] == ""
        assert fake._adv.footer_reasons[-1] == ""


# --------------------------------------------------------------------------- #
# 8. the console is cp1252-safe (the owner reads it on a Windows console)
# --------------------------------------------------------------------------- #
def test_the_new_methods_are_cp1252():
    """The two genuinely new methods, whole."""
    K = _K()
    for name in ("_reshape_commit_record", "_n_reshape_fragments"):
        for lineno, line in enumerate(
                inspect.getsource(getattr(K, name)).splitlines(), 1):
            try:
                line.encode("cp1252")
            except UnicodeEncodeError as exc:  # pragma: no cover
                pytest.fail("{} line {}: {}".format(name, lineno, exc))


def test_every_FRAG_diagnostic_is_cp1252():
    """The console the owner is told to copy is cp1252 on Windows, and a
    `[FRAG]` line that cannot be encoded takes the whole print with it.

    Scoped to the diagnostic lines rather than to whole methods:
    `_enter_fragment_refine_mode` and `_accept_split` predate this brick
    and carry non-ASCII arrows in their COMMENTS, which never reach a
    console."""
    import pathlib

    import napari_epicycle.annotator.curation_3d_widget as mod

    src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    seen = 0
    for lineno, line in enumerate(src.splitlines(), 1):
        if "[FRAG]" not in line:
            continue
        seen += 1
        try:
            line.encode("cp1252")
        except UnicodeEncodeError as exc:  # pragma: no cover
            pytest.fail("line {}: {}".format(lineno, exc))
    assert seen >= 4, "the brick promised [FRAG] at entry, accept and discard"

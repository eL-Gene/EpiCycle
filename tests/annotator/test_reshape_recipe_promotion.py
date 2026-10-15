"""Brick U1 -- what the recipe sends and bug B.

The rule, stated once: **the recipe sends the marker set the list shows.**

Three things follow, and each one is a bug in the version without it:

* source 0 and source 1 send ``paint`` with the UNTICKED ids zeroed out,
  so unticking a marker actually changes the cut;
* sources 2-4 send ``paint`` too, as of U1b: their Find materialises
  its markers onto the Refine layer, so the cut reads exactly the
  voxels the list describes.  The engine still runs the SAME detector
  inside ``find_seeds`` -- what changed is that the flood starts from
  those voxels instead of from a set the detector re-finds for itself.
  The merge-with-a-painted-addition promotion survives as the fallback
  for a materialise the ROI shape refused;
* source 1 ("Shape thickness peaks") showed the painting rows and then
  silently discarded everything painted, because the engine's
  ``distance`` detector never looks at the Refine layer.  Its Find now
  MATERIALISES the markers into that layer, so it becomes an ordinary
  painting source -- and since U1b so does every other found source.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


def _M():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


class FakeLayer:
    def __init__(self, data):
        self.data = data
        self.refreshed = 0
        self.selected_label = 0

    def refresh(self):
        self.refreshed += 1


class FakeAdv:
    def __init__(self, seeds_from=0, checked=None):
        self._seeds_from = seeds_from
        self._checked = checked
        self.calls = []

    def get_split_params(self):
        return {"seeds_from": self._seeds_from}

    def get_checked_marker_ids(self):
        return self._checked

    def update_split_ui_state(self, **kw):
        self.calls.append(kw)


class FakeSync:
    def __init__(self, shared):
        self.shared_labels = shared


class Fake:
    def __init__(self, roi_data, painted, seeds_from=0, checked=None,
                 target=7):
        shape = roi_data.shape
        self._split_roi = tuple(slice(0, n) for n in shape)
        self._refine_layer = FakeLayer(roi_data.copy())
        self._split_label_id = target
        self._split_running = True
        self._split_previewed = False
        self._split_painted_marker_ids = list(painted)
        self._split_seed_ids = list(painted)
        self._reshape_found_seeds = None
        self._active_labels_name = "curation_3d_x"
        self.sync_manager = FakeSync({"curation_3d_x": np.zeros(shape, np.int32)})
        self._adv = FakeAdv(seeds_from=seeds_from, checked=checked)
        self.subviewer_refreshes = 0

    def _get_advanced_features_widget(self):
        return self._adv

    def _refresh_refine_in_subviewers(self):
        self.subviewer_refreshes += 1


def _bind(fake, *names):
    K = _K()
    for n in names:
        setattr(fake, n, getattr(K, n).__get__(fake, Fake))
    return fake


# --------------------------------------------------------------------------- #
# the routing rules, read off the recipe's source
# --------------------------------------------------------------------------- #
class TestTheRecipeRules:
    def test_the_paint_branch_reads_the_PAINTED_ids(self):
        """Not `_split_seed_ids` -- a preview overwrites that with PIECE
        ids, so the next Preview would hand the engine the whole
        previous partition as seeds."""
        src = inspect.getsource(_K()._reshape_recipe)
        i_paint = src.index('elif seeds_kind == "paint":')
        tail = src[i_paint:i_paint + 2000]
        assert "_split_painted_marker_ids" in tail
        assert "self._split_seed_ids or []" not in tail

    def test_the_paint_branch_honours_the_ticks(self):
        src = inspect.getsource(_K()._reshape_recipe)
        i_paint = src.index('elif seeds_kind == "paint":')
        assert "_checked_marker_ids" in src[i_paint:i_paint + 2000]

    def test_a_found_source_becomes_a_painting_source_once_it_has_markers(
            self):
        """Bug B for index 1, widened to every found source by
        U1b."""
        src = inspect.getsource(_K()._reshape_recipe)
        assert "SEEDS_FROM_DETECTED" in src
        assert "seeds_from = SEEDS_FROM_PAINTED" in src
        assert "FIND_MARKER_SOURCES" in src

    def test_the_promotion_is_off_unless_the_user_paints(self):
        """It is the FALLBACK path now (U1b): the ordinary flow has
        already re-pointed a materialised source at `paint`, so this
        only fires when the materialise was refused."""
        src = inspect.getsource(_K()._reshape_recipe)
        i = src.index("promote_found = (")
        assert "painted_now" in src[i:i + 400]

    def test_a_FIND_never_sees_either_rule(self):
        """Pressing Find twice on source 1 would otherwise send `paint`
        the second time and find nothing new."""
        src = inspect.getsource(_K()._reshape_recipe)
        assert "for_find" in src
        find_src = inspect.getsource(_K()._find_reshape_markers)
        assert "for_find=True" in find_src

    def test_painted_now_ignores_an_id_with_no_voxels(self):
        """`_start_split` allocates the first id before anything is
        painted, so "the list is non-empty" is not "the user painted"."""
        roi = np.zeros((4, 4, 4), np.int32)
        fake = Fake(roi, painted=[101])
        _bind(fake, "_painted_marker_ids_on_layer")
        assert fake._painted_marker_ids_on_layer() == []
        fake._refine_layer.data[0, 0, 0] = 101
        assert fake._painted_marker_ids_on_layer() == [101]

    def test_painted_now_never_counts_the_target(self):
        roi = np.zeros((4, 4, 4), np.int32)
        roi[1:, :, :] = 7
        fake = Fake(roi, painted=[7, 101])
        _bind(fake, "_painted_marker_ids_on_layer")
        assert fake._painted_marker_ids_on_layer() == []


# --------------------------------------------------------------------------- #
# the merged seed layer (the promotion's payload)
# --------------------------------------------------------------------------- #
class TestTheMergedSeedLayer:
    def _f(self, checked=None):
        roi = np.zeros((4, 4, 4), np.int32)
        roi[3, 3, 3] = 101              # one hand-painted addition
        fake = Fake(roi, painted=[101], checked=checked)
        found = np.zeros((4, 4, 4), np.int32)
        found[0, 0, 0] = 1
        found[1, 1, 1] = 2
        fake._reshape_found_seeds = found
        _bind(fake, "_merged_marker_seed_layer", "_checked_marker_ids")
        return fake

    def test_both_sets_are_in_the_array(self):
        out = self._f()._merged_marker_seed_layer()
        ids = set(int(v) for v in np.unique(out)) - {0}
        assert len(ids) == 3, (
            "two found markers plus one painted one")

    def test_the_painted_id_is_kept_verbatim(self):
        out = self._f()._merged_marker_seed_layer()
        assert out[3, 3, 3] == 101

    def test_found_ids_are_offset_above_every_painted_one(self):
        """The detectors number from 1; the painted ids are minted from
        the volume's maximum.  The solver is invariant to the
        numbering, so offsetting is free and a collision is not."""
        out = self._f()._merged_marker_seed_layer()
        assert out[0, 0, 0] > 101
        assert out[1, 1, 1] > 101
        assert out[0, 0, 0] != out[1, 1, 1]

    def test_an_unticked_FOUND_marker_is_left_out(self):
        fake = self._f(checked=[2, 101])
        out = fake._merged_marker_seed_layer()
        assert out[0, 0, 0] == 0, "found marker 1 was unticked"
        assert out[1, 1, 1] != 0

    def test_an_unticked_PAINTED_marker_is_left_out(self):
        fake = self._f(checked=[1, 2])
        out = fake._merged_marker_seed_layer()
        assert out[3, 3, 3] == 0

    def test_nothing_to_merge_returns_None(self):
        fake = self._f()
        fake._reshape_found_seeds = None
        assert fake._merged_marker_seed_layer() is None

    def test_a_shape_mismatch_is_refused_rather_than_broadcast(self):
        fake = self._f()
        fake._reshape_found_seeds = np.zeros((2, 2, 2), np.int32)
        assert fake._merged_marker_seed_layer() is None


# --------------------------------------------------------------------------- #
# bug B -- materialising source 1's markers into the Refine layer
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self, seeds):
        self.seeds = seeds


class TestMaterialiseFoundMarkers:
    def _f(self):
        roi = np.zeros((4, 4, 4), np.int32)
        roi[1:, :, :] = 7            # the object
        roi[0, 0, 0] = 101           # one hand-painted marker
        fake = Fake(roi, painted=[101], seeds_from=1)
        _bind(fake, "_materialise_found_markers",
              "_drop_materialised_markers",
              "_marker_source_materialises", "_marker_source_index",
              # the materialiser drops ids with no voxels through it
              "_painted_marker_ids_on_layer")
        seeds = np.zeros((4, 4, 4), np.int32)
        seeds[2, 2, 2] = 1
        seeds[3, 3, 3] = 2
        markers = [{"id": 1, "voxels": 1, "centroid_zyx": (2, 2, 2)},
                   {"id": 2, "voxels": 1, "centroid_zyx": (3, 3, 3)}]
        return fake, Report(seeds), markers

    def test_every_FOUND_source_materialises(self):
        """U1b (2.5): it was index 1 alone, and the owner found the
        consequence -- sources 2-4 drew their markers in
        `guide_markers`, a MAIN-viewer layer, so the side views showed
        nothing.  Source 0 is the exception: its markers are already
        painted on the Refine layer, so there is nothing to move."""
        fake, _r, _m = self._f()
        for idx in (1, 2, 3, 4):
            fake._adv._seeds_from = idx
            assert fake._marker_source_materialises() is True, idx
        fake._adv._seeds_from = 0
        assert fake._marker_source_materialises() is False

    def test_the_marker_voxels_land_in_the_refine_layer(self):
        fake, report, markers = self._f()
        fake._materialise_found_markers(report, markers)
        data = fake._refine_layer.data
        assert data[2, 2, 2] not in (0, 7)
        assert data[3, 3, 3] not in (0, 7)

    def test_the_ids_are_minted_above_everything_live(self):
        """A-H4: a materialised marker must never collide with a painted
        one or with a live label."""
        fake, report, markers = self._f()
        fake.sync_manager.shared_labels["curation_3d_x"][0, 0, 1] = 500
        fake._materialise_found_markers(report, markers)
        data = fake._refine_layer.data
        assert data[2, 2, 2] > 500
        assert data[3, 3, 3] > 500
        assert data[2, 2, 2] != data[3, 3, 3]

    def test_the_hand_painted_marker_SURVIVES(self):
        """The reason it must not go through `_apply_seed_threshold`,
        which resets the ROI to `_refine_baseline` first."""
        fake, report, markers = self._f()
        fake._materialise_found_markers(report, markers)
        assert fake._refine_layer.data[0, 0, 0] == 101
        assert 101 in fake._split_painted_marker_ids

    def test_it_writes_the_report_voxels_VERBATIM(self):
        """Not spheres: `_apply_seed_threshold` inflates every peak with
        `sphere_radius=2`, which would hand the engine a different seed
        array from the one the report described."""
        fake, report, markers = self._f()
        fake._materialise_found_markers(report, markers)
        data = fake._refine_layer.data
        new = set(int(v) for v in np.unique(data)) - {0, 7, 101}
        for nid in new:
            assert int((data == nid).sum()) == 1

    def test_the_object_itself_is_untouched_elsewhere(self):
        fake, report, markers = self._f()
        before = (fake._refine_layer.data == 7).sum()
        fake._materialise_found_markers(report, markers)
        after = (fake._refine_layer.data == 7).sum()
        assert after == before - 2, "only the two marker voxels changed"

    def test_the_returned_list_is_every_marker_the_panel_shows(self):
        fake, report, markers = self._f()
        ids, stats = fake._materialise_found_markers(report, markers)
        assert ids == fake._split_painted_marker_ids
        assert 101 in ids and len(ids) == 3
        assert set(stats) <= set(ids)

    def test_the_stats_follow_the_new_numbering(self):
        fake, report, markers = self._f()
        ids, stats = fake._materialise_found_markers(report, markers)
        for nid, st in stats.items():
            assert nid in ids
            assert "centroid_zyx" in st

    def test_a_shape_mismatch_is_refused(self):
        fake, _report, markers = self._f()
        bad = Report(np.zeros((2, 2, 2), np.int32))
        before = fake._refine_layer.data.copy()
        fake._materialise_found_markers(bad, markers)
        assert np.array_equal(fake._refine_layer.data, before)

    def test_the_finish_handler_routes_a_found_source_through_it(self):
        src = inspect.getsource(_K()._on_seed_find_finished)
        assert "_marker_source_materialises" in src
        assert "_materialise_found_markers" in src
        assert "_show_marker_layer" not in src, (
            "U1b: there is no second layer left to fall back to -- every "
            "found source paints its markers on the Refine layer, and "
            "`guide_markers` is not created for markers at all")


class TestAPaintRecipeNeverAlsoSendsTheTicks:
    def test_a_promoted_recipe_clears_keep_seed_ids(self):
        """The promotion has ALREADY applied the ticks, and it
        RENUMBERS the found ids above the painted ones.  Sending the
        tick list on as well would run `np.isin(seeds, keep)` against
        ids that no longer exist and filter every promoted marker
        away."""
        src = inspect.getsource(_K()._preview_reshape)
        i_read = src.index("keep_seed_ids = adv.get_checked_marker_ids()")
        i_clear = src.index('getattr(self, "_reshape_promoted", False)')
        i_launch = src.index("keep_seed_ids=keep_seed_ids")
        assert i_read < i_clear < i_launch

    def test_it_is_NARROW__an_unpromoted_paint_still_sends_them(self):
        """An unpromoted `paint` recipe is the legacy shape: the ids are
        the painted ones, so the list is a no-op there and the call
        recorder in `test_reshape_split_orchestration.py` pins it."""
        src = inspect.getsource(_K()._preview_reshape)
        assert 'getattr(spec, "seeds", "") == "paint"' not in src

    def test_the_flag_is_reset_on_every_recipe_build(self):
        """A stale True would silently drop the tick list on the next,
        unpromoted cut."""
        src = inspect.getsource(_K()._reshape_recipe)
        i_reset = src.index("self._reshape_promoted = False")
        i_set = src.index("self._reshape_promoted = True")
        assert i_reset < i_set

    def test_the_all_unticked_refusal_still_fires_for_a_detector(self):
        """...and the clear must not swallow the "you unticked every
        marker" refusal on the sources that DO send the tick list."""
        src = inspect.getsource(_K()._preview_reshape)
        i_clear = src.index('getattr(self, "_reshape_promoted", False)')
        i_refuse = src.index("if keep_seed_ids is not None and not keep_seed_ids:")
        assert i_clear < i_refuse

    def test_unticking_every_PAINTED_marker_says_so(self):
        """It used to say "paint a marker first", which is a lie to
        someone who painted three and unticked them."""
        src = inspect.getsource(_K()._reshape_recipe)
        assert "You unticked every marker" in src


class TestAPaintedAdditionDoesNotReplaceTheFoundRows:
    def _f(self):
        roi = np.zeros((4, 4, 4), np.int32)
        fake = Fake(roi, painted=[101, 102])
        fake._reshape_found_marker_ids = [1, 2, 3]
        fake._reshape_found_marker_stats = {
            1: {"voxels": 9}, 2: {"voxels": 8}, 3: {"voxels": 7}}
        _bind(fake, "_marker_rows_for_panel")
        return fake

    def test_the_list_shows_both_sets(self):
        rows, stats = self._f()._marker_rows_for_panel()
        assert rows == [1, 2, 3, 101, 102]
        assert set(stats) == {1, 2, 3}

    def test_the_found_rows_keep_their_numbers(self):
        """Order is found-then-painted, so the row numbers a Find gave
        do not shuffle when a marker is added by hand."""
        rows, _ = self._f()._marker_rows_for_panel()
        assert rows[:3] == [1, 2, 3]

    def test_a_materialised_source_lists_each_marker_once(self):
        fake = self._f()
        fake._reshape_found_marker_ids = []     # source 1 clears them
        fake._split_painted_marker_ids = [101, 102, 103]
        rows, _ = fake._marker_rows_for_panel()
        assert rows == [101, 102, 103]

    @pytest.mark.parametrize("method", [
        "_next_seed", "_undo_last_seed", "_refine_seeds"])
    def test_the_three_pushes_go_through_it(self, method):
        src = inspect.getsource(getattr(_K(), method))
        assert "_marker_rows_for_panel()" in src, (
            "{} would REPLACE a found list with the painted ids "
            "alone".format(method))


class TestTheFindRecipeStillUsesTheDetector:
    def test_index_1_maps_to_distance_for_a_find(self):
        M = _M()
        src = inspect.getsource(_K()._reshape_recipe)
        assert 'SEEDS_FROM_DETECTED: "distance"' in src
        assert M.SEEDS_FROM_DETECTED == 1

    def test_index_1_is_in_the_find_set_and_not_the_detect_set(self):
        M = _M()
        assert M.SEEDS_FROM_DETECTED in M.FIND_MARKER_SOURCES
        assert M.SEEDS_FROM_DETECTED not in M.DETECT_SEED_SOURCES

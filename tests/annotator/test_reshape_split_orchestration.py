"""Reshape R1 -- the guided-Split orchestration, on Qt-free fakes.

The real `Curation3DAnnotator` methods are bound to a minimal fake with
`Method.__get__(fake)`, so these exercise the ACTUAL code rather than a
paraphrase of it.  That matters most for the two things a green suite would
otherwise miss:

* the legacy split path staying byte-identical (proved by a call recorder,
  not by reading the diff), and
* the target label being restored on EVERY exit -- between Start and Accept
  the cell does not exist in the volume, so a missed exit path loses it.
"""

from __future__ import annotations

import threading

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
    """A stand-in for the Split tab, with just the surface the widget reads."""

    def __init__(self, params=None, checked=None):
        M = _M()
        self._params = dict(
            seeds_from=M.SEEDS_FROM_PAINTED, walls_from=M.WALLS_FROM_SHAPE,
            guided=False, wall_channel="", clip=0.01, ridge=True,
            arbiter=False, arbiter_percentile=75.0, solver="watershed",
            rw_beta=130.0, compactness=0.0, h=0.15, min_distance_um=5.0,
            refine_guided=False, refine_merge_small=True, min_size=20,
            target_count=None,
            # Brick 2b: the eleven knobs the recipe now forwards.  Same
            # defaults as `reshape.spec.DEFAULT_PARAMS`.
            seed_smooth=2.0, max_sigma_um=6.0, seed_thresh=None,
            seed_clip=0.02, lift_mode="gentle", seed_gamma=0.5,
            seed_bg=15, seed_denoise=2, smooth=1.0,
            refine_radius=4, refine_eps=0.01,
        )
        self._params.update(params or {})
        self._checked = checked
        self.ui_calls = []
        self.cleared = 0
        self.sources = None
        self.channels = None
        self._marker_rows = []
        self._marker_off = set()

    def get_split_params(self):
        return dict(self._params)

    def is_guided_split(self):
        M = _M()
        return (self._params["seeds_from"] in M.IMAGE_DETECTOR_SOURCES
                or self._params["walls_from"] != M.WALLS_FROM_SHAPE)

    def guided_needs_guide_mask(self):
        M = _M()
        return self._params["seeds_from"] in M.GUIDED_SEED_SOURCES

    def get_checked_split_pieces(self):
        return list(self._checked) if self._checked is not None else []

    def update_split_guided_ui(self, **kw):
        self.ui_calls.append(kw)

    def update_split_ui_state(self, **kw):
        # Brick 2a: the marker picker's kwargs land here.
        self.ui_calls.append(kw)
        # U2b: the list is no longer write-only for this fake.  Parking
        # asks it which rows are OFF, and the whole H2 class of bug is
        # "the list was emptied and nobody reconciled", so a fake that
        # forgets what it was told could not see it.
        ids = kw.get("marker_ids")
        preserve_all = False
        if ids is None:
            ids = kw.get("seed_ids")
            preserve_all = ids is not None
        if ids is None:
            return
        spare = {int(i) for i in (kw.get("marker_preserve_ids") or [])}
        rows = [int(i) for i in ids]
        self._marker_rows = rows
        self._marker_off = {
            i for i in self._marker_off
            if i in set(rows) and (preserve_all or i in spare)
        }

    def set_marker_rows(self, rows, off=()):
        """Test helper: what the panel is showing right now."""
        self._marker_rows = [int(i) for i in rows]
        self._marker_off = {int(i) for i in off}

    def get_checked_marker_ids(self):
        if not self._marker_rows:
            return None
        return [i for i in self._marker_rows if i not in self._marker_off]

    def get_unticked_marker_ids(self):
        if not self._marker_rows:
            return None
        return [i for i in self._marker_rows if i in self._marker_off]

    def set_seed_channels(self, names, current=""):
        self.seed_channels = list(names)

    def clear_split_guided_ui(self):
        self.cleared += 1

    def set_guide_sources(self, sources):
        self.sources = list(sources)

    def set_wall_channels(self, names, current=""):
        self.channels = list(names)


class FakeAnnotator:
    """Only the attributes the reshape methods actually touch."""

    def __init__(self, volume, roi, target_id, adv=None, refine=None):
        self._active_labels_name = "curation_3d_x"
        self.sync_manager = FakeSync({"curation_3d_x": volume})
        self._split_roi = roi
        self._split_label_id = target_id
        self._split_snapshot = volume[roi].copy()
        self._split_running = True
        self._split_previewed = False
        self._split_seed_ids = []
        # Brick U1: the source of truth for the painted markers.  A
        # preview overwrites `_split_seed_ids` with PIECE ids, so it
        # cannot be it (UX round 1.2).
        self._split_painted_marker_ids = []
        self._refine_pre_preview = None
        self._reshape_found_seeds = None
        self._refine_layer = refine
        self._refine_baseline = None
        self._reshape_active = True
        self._reshape_running = False
        self._reshape_thread = None
        self._reshape_worker = None
        self._reshape_result = None
        self._reshape_final_ids = {}
        self._reshape_flags = {}
        self._reshape_guide = None
        self._reshape_guide_layer = None
        # --- Brick 2a ---
        self._temp_guide_layers = {}
        self._seed_find_running = False
        self._seed_find_thread = None
        self._seed_find_worker = None
        self._seed_find_token = 0
        self._seed_find_launch_token = None
        self._seed_find_launch_labels_name = "curation_3d_x"
        self._seed_find_launch_patch = "p"
        self._seed_find_launch_run_epoch = 0
        self._split_busy_kind = None
        self._reshape_guide_path = ""
        self._reshape_guide_run = ""
        self._reshape_override = False
        self._reshape_arbiter_merged = False
        self._reshape_token = 0
        self._reshape_launch_labels_name = "curation_3d_x"
        self._reshape_launch_patch = "p"
        self._reshape_launch_run_epoch = 0
        self._reshape_lock_fired = False
        self._guide_thread = None
        self._guide_worker = None
        self._guide_pending = None
        self._adv = adv or FakeAdv()
        # recorders
        self.lock_calls = []
        self.ui_calls = []
        self.synced = []
        self.next_id = 100
        self.subviewer_refreshes = 0
        self.restored = []
        self.removed = []

    # --- the seams the real methods call --------------------------------- #
    def _get_advanced_features_widget(self):
        return self._adv

    def _update_split_ui(self, **kw):
        self.ui_calls.append(kw)

    def _next_label_id_after_accept(self):
        return self.next_id

    def _sync_all_views_after_operation(self, roi=None, label_id=None, **kw):
        self.synced.append((roi, label_id))

    def _refresh_refine_in_subviewers(self):
        self.subviewer_refreshes += 1

    def _restore_ready_to_draw_state(self):
        pass

    def _lock_subviewers_for_preview(self):
        pass

    def _remove_edt_overlay(self):
        self.removed.append("edt")

    def _remove_refine_layer(self):
        self.removed.append("refine")

    def _reset_split_state(self):
        self._split_running = False
        self._reshape_active = False
        self._reshape_result = None

    def _layer_by_name(self, layers, name):
        return None

    def _reshape_context_cb(self):
        return {"workspace": "", "patch_name": "p", "run_id": None,
                "run_epoch": 0, "downsample_factor": 1,
                "spacing": (1.0, 1.0, 1.0)}

    def _reshape_lock_cb(self, active):
        self.lock_calls.append(bool(active))


#: Brick 2a added a temporary-layer registry and a second worker, and
#: several long-standing methods now call into them.  Binding them on
#: EVERY fake keeps the existing tests exercising the real code rather
#: than a paraphrase of it.
_ALWAYS_BOUND = (
    "_add_temp_layer", "_remove_temp_layer", "_remove_all_temp_layers",
    "_abandon_seed_find_worker", "_seed_find_thread_cleanup",
    # the stale-finish seam: the handlers read the FINISHING worker's
    # token through it, and a fake overrides `_signal_sender` only.
    "_result_token", "_signal_sender", "_reshape_context",
    # the routing predicate and its late-arm partner: `_preview_split`
    # reads the LIVE pickers now, not a flag latched at Start.  The arm
    # also writes the ready line, so its text helper comes with it.
    "_arm_guided_if_needed", "_is_guided_now", "_reshape_ready_text",
    # Brick U1: the loop's one exit, the adds-only reconciler and the
    # three seams the recipe reads the marker set through.
    "_leave_reshape_review", "_marker_rows_from_refine",
    "_painted_marker_ids_on_layer", "_checked_marker_ids",
    "_apply_marker_unticks", "_merged_marker_seed_layer",
    # Brick U2b: parking.  Bound on EVERY fake because the Find
    # handlers reconcile it now -- a fake that stubbed it out would
    # pass the H2 tests by not having the bug.
    "_sync_marker_parking", "_park_marker", "_unpark_marker",
    "_unpark_all_markers", "_parked_marker_ids", "_unticked_marker_ids",
    # Brick U4: the commit bookkeeping is a seam of its own now, because
    # fragment refine needs it WITHOUT the volume write.
    # `_write_reshape_result` delegates to it, so every fake that binds
    # the write needs the record too.
    "_reshape_commit_record", "_n_reshape_fragments",
    # Brick U3: the snap-back and the three exits drop the wall picture,
    # so every fake that binds one of them needs the REAL dropper --
    # a stub would pass by not having the wire.
    "_drop_walls_picture", "_show_walls_picture", "_walls_display_crop",
    # Brick B1: `_walls_display_crop` now asks WHICH tool
    # has a picture staged instead of reading `_reshape_result` directly,
    # so the two accessors are part of the same real wire.  A stub for
    # them would pass by not having it.
    "_active_elevation", "_active_display_roi",
    # Brick U3 fix: both `_preview_*` methods ask this before they
    # overwrite the marker snapshot, so a fake that drives either one
    # needs the real predicate.
    "_review_is_on_screen",
)


def bind(fake, *names):
    K = _K()
    for name in tuple(names) + _ALWAYS_BOUND:
        setattr(fake, name, getattr(K, name).__get__(fake, FakeAnnotator))
    return fake


# --------------------------------------------------------------------------- #
# a real engine result to commit
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def solved():
    """A genuine two-piece ReshapeResult from the real engine."""
    labels, guide, membrane, _ = two_columns(0.30)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    res = reshape_region(labels, {"membrane": membrane}, spec,
                         target_id=7, seed_layer=guide)
    assert res.status == "ok" and res.n_pieces >= 2
    return labels, res


# --------------------------------------------------------------------------- #
# 1. THE regression lock: the legacy path is untouched
# --------------------------------------------------------------------------- #
class TestLegacyPathUntouched:
    def test_preview_with_default_pickers_does_NOT_enter_the_guided_branch(self):
        """A call recorder, not a diff read: with `_reshape_active` False the
        guided entry point must never be reached, so the legacy body below it
        runs exactly as it always has -- byte-identical **while every marker
        is ticked**, which is the default.

        Brick U1 (UX round 2.3) changed the manual solver's INPUT, not its
        call: unticking a painted marker zeroes that id out of the ROI copy
        handed to `seeded_watershed_split`.  With nothing unticked the array
        is bit-identical (asserted by identity in
        `test_reshape_review_loop.py::TestUnticksReachTheSolver`)."""
        K = _K()
        fake = FakeAnnotator(np.zeros((4, 4, 4), np.int32),
                             (slice(0, 2),) * 3, 7)
        fake._reshape_active = False
        fake._refine_baseline = np.zeros((2, 2, 2), np.int32)
        fake._refine_layer = None
        called = []
        fake._preview_reshape = lambda: called.append("guided")
        bind(fake, "_preview_split")

        fake._preview_split()

        assert called == [], "the legacy preview must not route to the engine"
        # It fell through to the legacy body, which bails on a missing Refine
        # layer with its own message -- proof it reached the old code.
        assert any("Refine layer missing" in str(c.get("status_text", ""))
                   for c in fake.ui_calls)

    def test_preview_with_a_moved_picker_DOES_enter_the_guided_branch(self):
        M = _M()
        fake = FakeAnnotator(
            np.zeros((4, 4, 4), np.int32), (slice(0, 2),) * 3, 7,
            adv=FakeAdv({"walls_from": M.WALLS_FROM_MEMBRANE}))
        fake._reshape_active = True
        fake._refine_baseline = np.zeros((2, 2, 2), np.int32)
        called = []
        fake._preview_reshape = lambda: called.append("guided")
        bind(fake, "_preview_split")

        fake._preview_split()

        assert called == ["guided"]

    def test_the_guided_branch_is_the_FIRST_thing_after_the_existing_guards(self):
        """Placement matters: if it sat lower, the legacy body would already
        have run and the guided path would double-solve."""
        import inspect

        src = inspect.getsource(_K()._preview_split)
        i_guard = src.index("_refine_baseline is None")
        i_guided = src.index("_preview_reshape()")
        i_legacy = src.index("seeded_watershed_split")
        assert i_guard < i_guided < i_legacy

    def test_the_legacy_preview_snapshots_the_markers_before_it_solves(self):
        """The snap-back's source on this path (UX round 1.2).  It has
        to be taken BEFORE the unticks are applied, so it holds every
        painted marker -- an unticked marker stays DRAWN."""
        import inspect

        src = inspect.getsource(_K()._preview_split)
        i_snap = src.index("self._refine_pre_preview = current_roi.copy()")
        i_untick = src.index("_apply_marker_unticks")
        i_solve = src.index("seeded_watershed_split(")
        assert i_snap < i_untick < i_solve

    def test_a_legacy_accept_never_builds_reshape_bookkeeping(self):
        import inspect

        src = inspect.getsource(_K()._accept_split)
        # The guarded branch, and the legacy write still present beneath
        # it.  The guard is TWO conditions since 2026-09-19: the flag AND
        # the result the last preview left, because the flag alone can be
        # armed after a legacy preview and the engine branch then throws
        # that preview away (see TestAcceptCommitsWhatThePreviewProduced).
        assert 'getattr(self, "_reshape_active", False)' in src
        assert 'getattr(self, "_reshape_result", None) is not None' in src
        assert "write_mask = (refine_roi > 0) & (target_area == 0)" in src
        assert "_reshape_commit = None" in src


# --------------------------------------------------------------------------- #
# 2. the recipe: picker -> engine enums
# --------------------------------------------------------------------------- #
class TestRecipe:
    def _fake(self, **params):
        M = _M()
        vol = np.zeros((6, 6, 6), np.int32)
        vol[2:4, 2:4, 2:4] = 7
        roi = (slice(0, 6),) * 3
        refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
        fake = FakeAnnotator(vol, roi, 7, adv=FakeAdv(params), refine=refine)
        fake.viewer = type("V", (), {"layers": []})()
        bind(fake, "_reshape_recipe")
        return fake, M

    def test_painted_seeds_and_shape_walls_map_to_paint_plus_distance(self):
        fake, M = self._fake(seeds_from=0, walls_from=0)
        fake._split_seed_ids = [7, 101, 102]
        fake._refine_layer.data[0, 0, 0] = 101
        fake._refine_layer.data[5, 5, 5] = 102
        spec, images, seeds = fake._reshape_recipe()
        assert spec.seeds == "paint"
        assert spec.walls == "distance", (
            "shape-only must be the -EDT, which is what today's split floods")
        assert images == {}
        assert seeds is not None and set(np.unique(seeds)) == {0, 101, 102}

    def test_painted_seeds_exclude_the_TARGET_id_from_the_markers(self):
        """The target's own voxels on the Refine layer are the REGION, not a
        marker.  Counting them as one would silently add a phantom cell."""
        fake, M = self._fake()
        fake._split_seed_ids = [7, 101]
        fake._refine_layer.data[2:4, 2:4, 2:4] = 7      # the region
        fake._refine_layer.data[0, 0, 0] = 101          # the only marker
        _spec, _img, seeds = fake._reshape_recipe()
        assert 7 not in set(np.unique(seeds))

    def test_detected_seeds_map_to_the_geometry_blind_detector(self):
        fake, M = self._fake(seeds_from=1)
        spec, _i, seeds = fake._reshape_recipe()
        assert spec.seeds == "distance"
        assert seeds is None, "the engine finds these itself"

    def test_both_mask_sources_map_to_labels_seeds(self):
        M = _M()
        for idx in sorted(M.GUIDED_SEED_SOURCES):
            fake, _ = self._fake(seeds_from=idx)
            fake._reshape_guide = np.zeros((6, 6, 6), np.int32)
            spec, _i, seeds = fake._reshape_recipe()
            assert spec.seeds == "labels"
            assert seeds is fake._reshape_guide

    def test_a_mask_source_with_no_mask_loaded_refuses_in_plain_words(self):
        M = _M()
        fake, _ = self._fake(seeds_from=M.SEEDS_FROM_RUN_LABELS)
        spec, reason, _s = fake._reshape_recipe()
        assert spec is None
        assert "Load the mask" in reason
        for word in ("dtype", "None", "ndarray", "seed_layer"):
            assert word not in reason

    def test_membrane_walls_with_no_channel_picked_refuse_in_plain_words(self):
        M = _M()
        fake, _ = self._fake(walls_from=M.WALLS_FROM_MEMBRANE)
        spec, reason, _s = fake._reshape_recipe()
        assert spec is None
        assert "walls" in reason.lower()

    def test_no_painted_markers_refuses_and_NEVER_falls_through(self):
        """The sandbox's silent fallback to an unseeded watershed can shatter
        a columnar cell into stacked discs."""
        fake, _ = self._fake()
        fake._split_seed_ids = [7]
        spec, reason, _s = fake._reshape_recipe()
        assert spec is None
        assert "marker" in reason.lower()

    def test_the_arbiter_is_dropped_when_the_walls_are_shape_derived(self):
        """Belt-and-braces: the engine REFUSES that combination outright."""
        fake, _ = self._fake(walls_from=0, arbiter=True)
        fake._split_seed_ids = [7, 101]
        fake._refine_layer.data[0, 0, 0] = 101
        spec, _i, _s = fake._reshape_recipe()
        assert spec.arbiter is False

    def test_split_anyway_turns_the_arbiter_off_for_the_re_run(self):
        M = _M()
        fake, _ = self._fake(walls_from=M.WALLS_FROM_MEMBRANE, arbiter=True,
                             wall_channel="mem")
        fake._split_seed_ids = [7, 101]
        fake._refine_layer.data[0, 0, 0] = 101
        img = FakeLayer(np.zeros((6, 6, 6), np.float32), "mem")
        from napari.layers import Image as NapariImage
        img.__class__ = type("FakeImage", (NapariImage,), {})
        fake.viewer.layers = []
        fake._layer_by_name = lambda layers, name: None
        # channel missing -> refusal, so just assert the override flag path
        fake._reshape_override = True
        spec, reason, _s = fake._reshape_recipe()
        assert spec is None and "walls" in reason.lower()

    def test_a_zero_target_count_never_asks_for_merge_to_count(self):
        fake, _ = self._fake(target_count=None)
        fake._split_seed_ids = [7, 101]
        fake._refine_layer.data[0, 0, 0] = 101
        spec, _i, _s = fake._reshape_recipe()
        assert "merge_to_count" not in spec.refine


# --------------------------------------------------------------------------- #
# 3. the commit -- safe-write, unticked pieces, ids
# --------------------------------------------------------------------------- #
class TestCommit:
    def _commit_fake(self, solved, checked=None):
        labels, res = solved
        vol = labels.copy()
        neighbour_before = vol.copy()
        roi = res.roi
        # reproduce the hole `_start_split` punches
        vol[vol == 7] = 0
        fake = FakeAnnotator(vol, roi, 7, adv=FakeAdv(checked=checked))
        piece_ids = [int(r["id"]) for r in res.piece_stats]
        fake._reshape_result = res
        fake._reshape_final_ids = {p: 100 + i for i, p in enumerate(piece_ids)}
        bind(fake, "_write_reshape_result")
        return fake, vol, neighbour_before, res

    def test_all_ticked_writes_every_piece_as_a_new_cell(self, solved):
        n = len(solved[1].piece_stats)
        fake, vol, _nb, res = self._commit_fake(
            solved, checked=[100 + i for i in range(n)])
        out = fake._write_reshape_result(vol, 7)
        assert out["written"] > 0
        assert out["superseded"] is True
        assert 7 not in set(np.unique(vol))
        assert set(out["new_ids"]) <= set(np.unique(vol))

    def test_an_unticked_piece_RETURNS_TO_THE_TARGET_not_to_background(
            self, solved):
        """Unticking means 'not a separate cell', NEVER 'delete that tissue'.
        The cell's total voxel count is unchanged."""
        labels, res = solved
        total_before = int((labels == 7).sum())
        keep_one = [100]                                   # tick only piece 0
        fake, vol, _nb, _r = self._commit_fake(solved, checked=keep_one)
        fake._write_reshape_result(vol, 7)

        occupied = int(((vol == 7) | np.isin(vol, keep_one)).sum())
        assert occupied == total_before, (
            "unticking must not delete tissue -- those voxels belong to the "
            "cell again"
        )
        assert 7 in set(np.unique(vol)), "the original cell must survive"

    def test_an_unticked_piece_leaves_the_original_id_NOT_superseded(
            self, solved):
        fake, vol, _nb, _r = self._commit_fake(solved, checked=[100])
        out = fake._write_reshape_result(vol, 7)
        assert out["superseded"] is False, (
            "the split callback would have marked it superseded, which would "
            "be a lie -- this is why reshape needs its own callback"
        )
        assert 7 not in out["new_ids"]

    def test_the_write_NEVER_touches_a_neighbour_label(self, solved):
        labels, res = solved
        vol = labels.copy()
        vol[0, 0, 0] = 999                       # a neighbour inside the ROI
        vol[vol == 7] = 0
        piece_ids = [int(r["id"]) for r in res.piece_stats]
        fake = FakeAnnotator(vol, res.roi, 7, adv=FakeAdv(
            checked=[100 + i for i in range(len(piece_ids))]))
        fake._reshape_result = res
        fake._reshape_final_ids = {p: 100 + i for i, p in enumerate(piece_ids)}
        bind(fake, "_write_reshape_result")

        fake._write_reshape_result(vol, 7)

        assert vol[0, 0, 0] == 999, "a neighbour label was overwritten"

    def test_piece_stats_reach_the_controller_in_VOLUME_coordinates(
            self, solved):
        labels, res = solved
        fake, vol, _nb, _r = self._commit_fake(
            solved, checked=[100 + i for i in range(len(res.piece_stats))])
        out = fake._write_reshape_result(vol, 7)
        assert out["piece_stats"]
        for row in out["piece_stats"]:
            z, y, x = row["centroid_zyx"]
            assert 0 <= z < labels.shape[0]
            assert 0 <= y < labels.shape[1]
            assert 0 <= x < labels.shape[2]
            assert row["voxels"] > 0
            assert row["id"] in out["new_ids"]

    def test_split_override_marks_every_result_row(self, solved):
        fake, vol, _nb, res = self._commit_fake(
            solved, checked=[100 + i for i in range(len(solved[1].piece_stats))])
        fake._reshape_override = True
        out = fake._write_reshape_result(vol, 7)
        assert out["flags"]
        assert set(out["flags"].values()) == {"split-override"}
        assert set(out["flags"]) == set(out["new_ids"])

    def test_a_no_new_ids_commit_still_records_a_dividing_verdict(self, solved):
        """The arbiter's most important answer is a NO-OP WRITE.  If the flag
        were dropped here there would be no undo entry and no record at all."""
        fake, vol, _nb, _r = self._commit_fake(solved, checked=[])
        out = fake._write_reshape_result(vol, 7)
        assert out["new_ids"] == []
        assert out["superseded"] is False
        assert out["flags"].get(7) == "dividing"

    def test_the_commit_carries_the_guide_run_for_provenance(self, solved):
        fake, vol, _nb, _r = self._commit_fake(solved, checked=[100])
        fake._reshape_guide_run = "NUCLEI-v2"
        out = fake._write_reshape_result(vol, 7)
        assert out["guide_run"] == "NUCLEI-v2"

    def test_no_result_means_no_commit(self):
        fake = FakeAnnotator(np.zeros((4, 4, 4), np.int32),
                             (slice(0, 4),) * 3, 7)
        bind(fake, "_write_reshape_result")
        assert fake._write_reshape_result(fake.sync_manager
                                          .shared_labels["curation_3d_x"], 7) is None


# --------------------------------------------------------------------------- #
# 4. ids are minted through the EXISTING helper
# --------------------------------------------------------------------------- #
class TestIdMinting:
    def test_final_ids_come_from_next_label_id_after_accept(self, solved):
        labels, res = solved
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        fake = FakeAnnotator(labels.copy(), res.roi, 7, refine=refine)
        fake.next_id = 512
        bind(fake, "_apply_reshape_result",
         "_print_count_diag_result", "_render_reshape_preview",
             "_reshape_quality_notes", "_reshape_review_status")

        fake._apply_reshape_result(res)

        assert min(fake._reshape_final_ids.values()) == 512
        # Brick U1: `_split_seed_ids` is NOT overwritten with the piece
        # ids any more -- that is what made the loop impossible (UX
        # round 1.2).  `_reshape_final_ids` is where the pieces live.
        assert fake._split_seed_ids == []
        assert sorted(fake._reshape_final_ids.values()) == [512, 513]

    def test_ids_are_consecutive_and_unique(self, solved):
        labels, res = solved
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        fake = FakeAnnotator(labels.copy(), res.roi, 7, refine=refine)
        fake.next_id = 40
        bind(fake, "_apply_reshape_result",
         "_print_count_diag_result", "_render_reshape_preview",
             "_reshape_quality_notes", "_reshape_review_status")
        fake._apply_reshape_result(res)
        vals = sorted(fake._reshape_final_ids.values())
        assert vals == list(range(40, 40 + len(vals)))


# --------------------------------------------------------------------------- #
# 5. the preview render -- ticks, no re-solve
# --------------------------------------------------------------------------- #
class TestPreviewRender:
    def _r(self, solved, checked):
        labels, res = solved
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        piece_ids = [int(x["id"]) for x in res.piece_stats]
        fake = FakeAnnotator(labels.copy(), res.roi, 7,
                             adv=FakeAdv(checked=checked), refine=refine)
        fake._reshape_result = res
        fake._reshape_final_ids = {p: 100 + i for i, p in enumerate(piece_ids)}
        bind(fake, "_render_reshape_preview", "_on_reshape_pieces_changed")
        return fake, refine

    def test_ticked_pieces_are_drawn_with_their_new_ids(self, solved):
        n = len(solved[1].piece_stats)
        fake, refine = self._r(solved, checked=[100 + i for i in range(n)])
        fake._render_reshape_preview()
        drawn = set(np.unique(refine.data)) - {0}
        assert drawn == {100 + i for i in range(n)}

    def test_an_unticked_piece_is_drawn_with_the_TARGET_id(self, solved):
        """WYSIWYG: the preview shows exactly where those voxels will go."""
        fake, refine = self._r(solved, checked=[100])
        fake._render_reshape_preview()
        drawn = set(np.unique(refine.data)) - {0}
        assert 7 in drawn, "the unticked piece must show as part of the cell"
        assert 100 in drawn

    def test_a_tick_change_redraws_and_MIRRORS_but_never_re_solves(self, solved):
        fake, refine = self._r(solved, checked=[100])
        before = fake.subviewer_refreshes
        fake._preview_reshape = lambda: pytest.fail("a tick must not re-solve")
        fake._on_reshape_pieces_changed()
        assert refine.refreshed > 0
        assert fake.subviewer_refreshes == before + 1, (
            "the preview must be mirrored into all three subviewers")

    def test_a_tick_change_with_no_result_is_a_no_op(self, solved):
        fake, _refine = self._r(solved, checked=[100])
        fake._reshape_result = None
        fake._on_reshape_pieces_changed()   # must not raise

    def test_a_tick_change_on_a_LEGACY_split_is_a_no_op(self, solved):
        fake, refine = self._r(solved, checked=[100])
        fake._reshape_active = False
        before = refine.refreshed
        fake._on_reshape_pieces_changed()
        assert refine.refreshed == before


# --------------------------------------------------------------------------- #
# 6. every exit restores the target (one test each)
# --------------------------------------------------------------------------- #
class TestEveryExitRestoresTheTarget:
    def _holed(self):
        labels, _guide, _mem, _r = two_columns(0.30)
        vol = labels.copy()
        roi = (slice(0, vol.shape[0]), slice(0, vol.shape[1]),
               slice(0, vol.shape[2]))
        snap = vol[roi].copy()
        vol[vol == 7] = 0                     # the hole `_start_split` punches
        fake = FakeAnnotator(vol, roi, 7)
        fake._split_snapshot = snap
        return fake, vol, snap

    def test_restore_puts_every_voxel_back_exactly(self):
        fake, vol, snap = self._holed()
        bind(fake, "_restore_reshape_target")
        fake._restore_reshape_target("test")
        assert np.array_equal(vol, snap)
        assert fake.synced == [(fake._split_roi, 7)]

    def test_teardown_on_patch_switch_restores_and_releases_the_lock(self):
        fake, vol, snap = self._holed()
        bind(fake, "_restore_reshape_target", "_teardown_reshape_for_switch",
             "_fire_reshape_lock", "_remove_guide_layer",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_abandon_reshape_worker")
        fake._reshape_lock_fired = True
        fake._teardown_reshape_for_switch()
        assert np.array_equal(vol, snap)
        assert fake.lock_calls == [False]
        assert fake._split_running is False

    def test_teardown_restores_a_LEGACY_session_too(self):
        """M1.  The gate was `_reshape_active and _split_running`, and a
        manual (shape-only) split has `_reshape_active` False -- while
        `_start_split` punches the object out for BOTH paths.  So a
        manual split open at a patch switch or at annotator close left
        the object missing from the volume, with its session state
        stranded on a widget about to be destroyed."""
        fake, vol, snap = self._holed()
        fake._reshape_active = False          # the legacy path
        bind(fake, "_restore_reshape_target", "_teardown_reshape_for_switch",
             "_fire_reshape_lock", "_remove_guide_layer",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_abandon_reshape_worker")
        assert 7 not in set(np.unique(vol)), "the hole must be real"
        fake._teardown_reshape_for_switch()
        assert np.array_equal(vol, snap)
        assert fake._split_running is False

    def test_teardown_always_empties_the_parked_stash(self):
        """It holds ROI coordinates, and the ROI belongs to the session
        that is ending -- carried forward it would paint one object's
        marker onto the next one (M1)."""
        fake, _vol, _snap = self._holed()
        fake._split_running = False           # nothing to restore
        fake._parked_marker_voxels = {101: ()}
        bind(fake, "_restore_reshape_target", "_teardown_reshape_for_switch",
             "_fire_reshape_lock", "_remove_guide_layer",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_abandon_reshape_worker")
        fake._teardown_reshape_for_switch()
        assert fake._parked_marker_voxels == {}

    def test_teardown_cancels_a_worker_in_flight(self):
        fake, _vol, _snap = self._holed()
        cancelled = []

        class W:
            def cancel(self_inner):
                cancelled.append(1)

        fake._reshape_running = True
        fake._reshape_worker = W()
        bind(fake, "_restore_reshape_target", "_teardown_reshape_for_switch",
             "_fire_reshape_lock", "_remove_guide_layer",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_abandon_reshape_worker")
        fake._teardown_reshape_for_switch()
        assert cancelled == [1]
        assert fake._reshape_running is False

    def test_teardown_is_idempotent(self):
        fake, vol, snap = self._holed()
        bind(fake, "_restore_reshape_target", "_teardown_reshape_for_switch",
             "_fire_reshape_lock", "_remove_guide_layer",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_abandon_reshape_worker")
        fake._teardown_reshape_for_switch()
        fake._teardown_reshape_for_switch()
        assert np.array_equal(vol, snap)

    def test_teardown_never_raises_even_when_everything_is_missing(self):
        fake = FakeAnnotator(np.zeros((2, 2, 2), np.int32),
                             (slice(0, 2),) * 3, 7)
        fake._split_snapshot = None
        fake.sync_manager.shared_labels = {}
        bind(fake, "_restore_reshape_target", "_teardown_reshape_for_switch",
             "_fire_reshape_lock", "_remove_guide_layer",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_abandon_reshape_worker")
        fake._teardown_reshape_for_switch()      # must not raise
        assert fake._reshape_lock_fired is False

    def test_accept_with_no_result_restores_rather_than_leaving_a_hole(self):
        """Drives the REAL restore, so stubbing it out fails this.

        (The source-inspection version this replaces passed whether or not
        the restore did anything -- AUDIT 2026-09-09.)"""
        fake, vol, snap = self._holed()
        bind(fake, "_restore_reshape_target")
        assert 7 not in set(np.unique(vol)), "the hole must be real"
        fake._restore_reshape_target("accept found no result")
        assert np.array_equal(vol, snap)

    def test_reject_releases_the_lock_and_clears_the_piece_list(self):
        """Behavioural: the real `_reject_split` runs and both effects are
        observed on the fakes (AUDIT 2026-09-09 -- was source inspection)."""
        labels, _g, _m, _r = two_columns(0.30)
        vol = labels.copy()
        roi = (slice(0, vol.shape[0]), slice(0, vol.shape[1]),
               slice(0, vol.shape[2]))
        snap = vol[roi].copy()
        vol[vol == 7] = 0
        f = _RealRejectFake(vol, roi, 7)
        f.sync_manager.shared_labels["curation_3d_x"] = vol
        f._split_snapshot = snap
        f._refine_layer = FakeLayer(np.zeros_like(vol), "Refine")
        f._reshape_lock_fired = True
        bind(f, "_reject_split", "_restore_reshape_target",
             "_fire_reshape_lock", "_reset_split_state",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_remove_guide_layer", "_teardown_reshape_for_switch",
             "_abandon_reshape_worker")
        f._reject_split()
        assert f.lock_calls == [False]
        assert f._adv.cleared == 1, "the piece list must be cleared"
        assert np.array_equal(vol, snap)

    def test_close_cleanup_tears_the_guided_session_down(self):
        import inspect

        src = inspect.getsource(_K()._close_cleanup)
        assert "_teardown_reshape_for_switch" in src

    def test_a_refused_result_leaves_the_volume_alone(self, solved):
        """Zero seeds: nothing is written, the hole stays until the user
        cancels, and the panel says what to do."""
        fake, vol, snap = self._holed()
        before = vol.copy()
        bind(fake, "_on_reshape_finished", "_reshape_is_stale",
             "_reshape_thread_cleanup", "_reshape_context")
        fake._reshape_running = True

        class R:
            status = "refused"
            messages = ["no seeds found in this cell - lower the prominence"]

        fake._on_reshape_finished(R())
        assert np.array_equal(vol, before)
        assert any("no seeds" in str(c.get("status_text", ""))
                   for c in fake.ui_calls)


# --------------------------------------------------------------------------- #
# 7. stale finish -- a lock is not a stale-finish guard
# --------------------------------------------------------------------------- #
class TestStaleFinish:
    def _fake(self):
        fake = FakeAnnotator(np.zeros((4, 4, 4), np.int32),
                             (slice(0, 4),) * 3, 7)
        bind(fake, "_reshape_is_stale", "_reshape_context",
             "_on_reshape_finished", "_reshape_thread_cleanup")
        fake._reshape_running = True
        fake._applied = []
        fake._apply_reshape_result = lambda r: fake._applied.append(r)
        return fake

    def test_a_result_for_a_different_patch_is_discarded(self):
        fake = self._fake()
        fake._reshape_launch_patch = "OTHER-PATCH"
        assert fake._reshape_is_stale() is True

    def test_a_result_after_a_run_switch_is_discarded(self):
        fake = self._fake()
        fake._reshape_launch_run_epoch = 99
        assert fake._reshape_is_stale() is True

    def test_a_result_after_the_labels_layer_changed_is_discarded(self):
        fake = self._fake()
        fake._reshape_launch_labels_name = "curation_3d_OTHER"
        assert fake._reshape_is_stale() is True

    def test_a_result_after_the_split_session_ended_is_discarded(self):
        fake = self._fake()
        fake._split_running = False
        assert fake._reshape_is_stale() is True

    def test_a_matching_result_is_NOT_stale(self):
        assert self._fake()._reshape_is_stale() is False

    def test_a_stale_finish_never_reaches_apply(self, solved):
        fake = self._fake()
        fake._reshape_launch_patch = "OTHER"
        fake._on_reshape_finished(solved[1])
        assert fake._applied == []

    def test_a_finish_after_CANCEL_is_discarded(self, solved):
        """Cancel means abandon the result -- the worker's eventual output
        must not land after the user has been told nothing changed."""
        fake = self._fake()
        fake._reshape_running = False           # cancel already cleared it
        fake._on_reshape_finished(solved[1])
        assert fake._applied == []


# --------------------------------------------------------------------------- #
# 8. the lock
# --------------------------------------------------------------------------- #
class TestLock:
    def test_the_lock_goes_on_BEFORE_the_hole_is_punched(self):
        """Order, not presence.  While the hole is open, Save Curation and
        Commit Triage must be unavailable or they will persist a volume with
        a hole in it."""
        import inspect

        src = inspect.getsource(_K()._start_split)
        i_lock = src.index("_fire_reshape_lock(True)")
        i_punch = src.index("shared_data[label_mask] = 0")
        i_snapshot = src.index("self._snapshot_label_roi(")
        assert i_lock < i_punch, "the lock must precede the hole punch"
        assert i_lock > src.index("if not label_mask.any():"), (
            "and it must sit AFTER every validation early-return, or a return "
            "would strand the whole Step 3 surface locked"
        )
        assert i_snapshot < i_punch

    def test_fire_lock_is_idempotent(self):
        fake = FakeAnnotator(np.zeros((2, 2, 2), np.int32),
                             (slice(0, 2),) * 3, 7)
        bind(fake, "_fire_reshape_lock")
        fake._fire_reshape_lock(True)
        fake._fire_reshape_lock(True)
        fake._fire_reshape_lock(False)
        fake._fire_reshape_lock(False)
        assert fake.lock_calls == [True, False]

    def test_a_raising_lock_callback_never_propagates(self):
        fake = FakeAnnotator(np.zeros((2, 2, 2), np.int32),
                             (slice(0, 2),) * 3, 7)
        fake._reshape_lock_cb = lambda a: (_ for _ in ()).throw(RuntimeError("x"))
        bind(fake, "_fire_reshape_lock")
        fake._fire_reshape_lock(True)          # must not raise

    def test_accept_releases_the_lock_last(self):
        import inspect

        src = inspect.getsource(_K()._accept_split)
        i_visible = src.index('main_layer.visible = True')
        i_release = src.rindex("_fire_reshape_lock(False)")
        assert i_release > i_visible, (
            "release only once the surface is whole again")

    def test_a_worker_launch_failure_does_not_strand_the_lock(self):
        import inspect

        src = inspect.getsource(_K()._preview_reshape)
        assert "except Exception as exc:" in src
        # The LAST handler is the launch one.  Brick 2a added an earlier
        # `except Exception` around the cherry-pick read, so splitting on
        # the first occurrence stopped looking at the right block.
        assert "_reshape_running = False" in src.split(
            "except Exception as exc:")[-1]

    def test_reshape_is_inside_the_navigator_isolation_gate(self):
        import inspect

        src = inspect.getsource(_K()._refinement_op_active)
        assert "_reshape_running" in src


# --------------------------------------------------------------------------- #
# 9. the guide layer -- unmanaged, read-only, swept
# --------------------------------------------------------------------------- #
class TestGuideLayerInvariants:
    def test_the_guide_is_marked_temporary_read_only_and_brush_locked(self):
        """Asserts the flags on the LAYER OBJECT the method produced, not on
        its source text (AUDIT 2026-09-09).

        Read back off the VIEWER, because the registry builds the layer
        itself and appends it now (Brick U3 fix pass, H2) -- so "it is
        flagged" and "it reached the viewer" are the same assertion.
        """
        f = FakeAnnotator(np.zeros((2, 2, 2), np.int32), (slice(0, 2),) * 3, 7)

        class _V:
            def __init__(self_v):
                self_v.layers = []

        f.viewer = _V()
        f._adv = None
        bind(f, "_add_guide_layer")
        f._add_guide_layer(np.zeros((2, 2, 2), np.int32), "guide_x_p")
        assert len(f.viewer.layers) == 1
        lyr = f.viewer.layers[-1]
        assert getattr(lyr, "_epicycle_temporary", False) is True
        assert lyr.editable is False
        assert int(lyr.brush_size) == 1, "brush lock"
        assert f._reshape_guide_layer is lyr

    def test_the_guide_never_enters_shared_labels_or_the_mirror_set(self):
        """Checked against the CODE, not the docstring -- the docstring names
        every one of these precisely to explain why they are absent."""
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(_K()._add_guide_layer))
        tree = ast.parse(src).body[0]
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)):
            tree.body = tree.body[1:]           # drop the docstring
        code = ast.unparse(tree)
        for forbidden in (
            "shared_labels", "add_labels_to_subviewers",
            "_active_labels_names", "_managed_layer_names",
        ):
            assert forbidden not in code, (
                "the guide must participate in nothing: found {!r}".format(
                    forbidden)
            )

    def test_the_guide_hides_before_it_is_removed(self):
        """Brick 2a: the removal itself moved into the registry, which is
        where the rule now has to hold -- for the mask AND the markers."""
        import inspect

        src = inspect.getsource(_K()._remove_temp_layer)
        i_hide = src.index("visible = False")
        i_remove = src.index("layers.remove")
        assert i_hide < i_remove, "freed-VBO rule"

    def test_the_guide_layer_name_carries_the_sweep_prefix(self):
        from napari_epicycle.annotator import reshape_guide as RG

        class S:
            kind = "run"
            run_id = "NUCLEI v2"
            display_name = "Nuclei"
            path = ""

        name = RG.guide_layer_name("patch 6.tiff", S())
        assert name.startswith("guide_"), (
            "the controller's _clear_curation_layers sweep keys on this prefix"
        )

    def test_remove_is_safe_when_there_is_no_guide(self):
        fake = FakeAnnotator(np.zeros((2, 2, 2), np.int32),
                             (slice(0, 2),) * 3, 7)
        bind(fake, "_remove_guide_layer")
        fake._remove_guide_layer()             # must not raise
        assert fake._reshape_guide is None

    def test_show_hide_is_a_plain_visibility_toggle(self):
        fake = FakeAnnotator(np.zeros((2, 2, 2), np.int32),
                             (slice(0, 2),) * 3, 7)
        layer = FakeLayer(np.zeros((2, 2, 2)), "guide_x_p")
        fake._reshape_guide_layer = layer
        bind(fake, "_on_reshape_guide_show")
        fake._on_reshape_guide_show(False)
        assert layer.visible is False
        fake._on_reshape_guide_show(True)
        assert layer.visible is True


# --------------------------------------------------------------------------- #
# 10. cancel
# --------------------------------------------------------------------------- #
class TestCancel:
    def test_cancel_sets_the_event_and_says_the_cell_is_still_held_out(self):
        fake = FakeAnnotator(np.zeros((4, 4, 4), np.int32),
                             (slice(0, 4),) * 3, 7)
        fake._reshape_parked_threads = []
        event = threading.Event()

        class W:
            def cancel(self_inner):
                event.set()

        fake._reshape_running = True
        fake._reshape_worker = W()
        bind(fake, "_on_reshape_cancel", "_abandon_reshape_worker",
             "_reshape_thread_cleanup")
        fake._on_reshape_cancel()
        assert event.is_set()
        assert fake._reshape_running is False
        # Stop abandons the WORKER; it does NOT put the cell back -- only
        # the footer's secondary does ("Stop is not
        # Discard").  The old wording here, "nothing was changed", was true
        # of the RESULT and false of the volume, which is the confusion the
        # two-button split exists to remove.
        texts = [str(c.get("status_text", "")) for c in fake.ui_calls]
        assert any("still held out" in t for t in texts), texts
        assert any("Discard" in t for t in texts), texts
        assert not any("nothing was changed" in t for t in texts), texts

    def test_cancel_with_no_worker_is_safe(self):
        fake = FakeAnnotator(np.zeros((4, 4, 4), np.int32),
                             (slice(0, 4),) * 3, 7)
        fake._reshape_parked_threads = []
        bind(fake, "_on_reshape_cancel", "_abandon_reshape_worker",
             "_reshape_thread_cleanup")
        fake._on_reshape_cancel()              # must not raise


# --------------------------------------------------------------------------- #
# 11. diagnostics stay console-safe
# --------------------------------------------------------------------------- #
def test_reshape_diagnostics_are_cp1252_safe():
    """The Windows console is cp1252; an arrow in a print takes the app down
    on the exact path a user is trying to report."""
    import inspect

    K = _K()
    for name in (
        "_fire_reshape_lock", "_reshape_context", "_refresh_reshape_inputs",
        "_on_reshape_guide_load", "_add_guide_layer", "_remove_guide_layer",
        "_reshape_recipe", "_preview_reshape", "_on_reshape_finished",
        "_apply_reshape_result", "_print_count_diag_result",
        "_render_reshape_preview",
        "_reshape_quality_notes", "_reshape_review_status",
        "_write_reshape_result", "_restore_reshape_target",
        "_teardown_reshape_for_switch", "_on_reshape_cancel",
        "_on_reshape_error", "_reshape_is_stale", "_reshape_ready_text",
    ):
        for lineno, line in enumerate(
                inspect.getsource(getattr(K, name)).splitlines(), 1):
            try:
                line.encode("cp1252")
            except UnicodeEncodeError as exc:  # pragma: no cover
                pytest.fail("{} line {}: {}".format(name, lineno, exc))


# --------------------------------------------------------------------------- #
# 12. a worker that will not stop must be PARKED, never dropped
#     (found live 2026-09-08: dropping it aborted the whole process)
# --------------------------------------------------------------------------- #
class _StubSignal:
    def __init__(self):
        self.disconnected = 0
        self.slots = []

    def disconnect(self):
        self.disconnected += 1

    def connect(self, slot):
        self.slots.append(slot)

    def fire(self):
        for s in list(self.slots):
            s()


class _StubThread:
    def __init__(self, stops):
        self._stops = stops
        self.quit_calls = 0
        self.finished = _StubSignal()

    def quit(self):
        self.quit_calls += 1

    def wait(self, ms):
        return self._stops


class _StubWorker:
    def __init__(self):
        for name in ("progress", "message", "finished", "error",
                     "refused"):
            setattr(self, name, _StubSignal())


class TestUnstoppableWorkerIsParked:
    def _fake(self):
        fake = FakeAnnotator(np.zeros((2, 2, 2), np.int32),
                             (slice(0, 2),) * 3, 7)
        fake._reshape_parked_threads = []
        bind(fake, "_reshape_thread_cleanup", "_guide_thread_cleanup")
        return fake

    def test_a_thread_that_stops_is_simply_released(self):
        fake = self._fake()
        th, w = _StubThread(True), _StubWorker()
        fake._reshape_thread, fake._reshape_worker = th, w
        fake._reshape_thread_cleanup()
        assert th.quit_calls == 1
        assert fake._reshape_parked_threads == []
        assert fake._reshape_thread is None

    def test_a_thread_that_will_NOT_stop_is_HELD_not_dropped(self):
        """Dropping it garbage-collects a running QThread, which aborts the
        process with "QThread: Destroyed while thread is still running".
        Observed live before this existed."""
        fake = self._fake()
        th, w = _StubThread(False), _StubWorker()
        fake._reshape_thread, fake._reshape_worker = th, w
        fake._reshape_thread_cleanup()
        assert len(fake._reshape_parked_threads) == 1
        held_th, held_w = fake._reshape_parked_threads[0]
        assert held_th is th and held_w is w
        assert fake._reshape_thread is None, (
            "the ACTIVE slot must be cleared so a new preview can start")

    def test_a_parked_worker_has_its_signals_disconnected(self):
        """Its answer is already being abandoned; it must not land on a widget
        that may be half torn down by the time it finishes."""
        fake = self._fake()
        th, w = _StubThread(False), _StubWorker()
        fake._reshape_thread, fake._reshape_worker = th, w
        fake._reshape_thread_cleanup()
        for name in ("progress", "message", "finished", "error"):
            assert getattr(w, name).disconnected == 1, name

    def test_a_parked_thread_releases_itself_when_it_finishes(self):
        fake = self._fake()
        th, w = _StubThread(False), _StubWorker()
        fake._reshape_thread, fake._reshape_worker = th, w
        fake._reshape_thread_cleanup()
        assert len(fake._reshape_parked_threads) == 1
        th.finished.fire()
        assert fake._reshape_parked_threads == []

    def test_the_GUIDE_reader_is_parked_the_same_way(self):
        fake = self._fake()
        th, w = _StubThread(False), _StubWorker()
        fake._guide_thread, fake._guide_worker = th, w
        fake._guide_thread_cleanup()
        assert len(fake._reshape_parked_threads) == 1
        assert fake._guide_thread is None
        for name in ("finished", "refused", "error"):
            assert getattr(w, name).disconnected == 1, name

    def test_cleanup_with_no_thread_is_a_no_op(self):
        fake = self._fake()
        fake._reshape_thread_cleanup()
        fake._guide_thread_cleanup()
        assert fake._reshape_parked_threads == []


# --------------------------------------------------------------------------- #
# 13. the navigator's isolation gate must survive a bare widget
# --------------------------------------------------------------------------- #
class TestFlagsResolveWithoutAnInstance:
    """`_refinement_op_active` is read by the navigator's isolation gate, and
    PyQt6 raises RuntimeError -- NOT AttributeError -- for a missing attribute
    on a QObject whose C++ side was never constructed.  So
    `getattr(self, "_reshape_running", False)` does NOT fall back to its
    default on such an object, and adding that term to the predicate broke
    four `test_isolation_op_guard` tests that build the widget with `__new__`.

    Class-level defaults resolve on the TYPE and never reach that machinery.
    """

    def test_the_reshape_flags_are_class_attributes(self):
        K = _K()
        assert K._reshape_running is False
        assert K._reshape_active is False

    def test_the_predicate_works_on_a_widget_with_no_init(self):
        K = _K()
        ann = K.__new__(K)
        ann._split_running = False
        ann._smooth_running = False
        ann._interp_snapshot = None
        ann._interp_refine_mode = False
        assert K._refinement_op_active(ann) is False

    def test_the_predicate_still_reports_a_running_reshape(self):
        K = _K()
        ann = K.__new__(K)
        ann._split_running = False
        ann._smooth_running = False
        ann._interp_snapshot = None
        ann._interp_refine_mode = False
        ann._reshape_running = True
        assert K._refinement_op_active(ann) is True

# =========================================================================== #
# LIFECYCLE AUDIT (2026-09-09) -- the punched-out-cell window.
#
# Between Start and Accept the cell does NOT exist in the volume.  Every path
# out of that window must either write it or put it back, and none may leave a
# worker running, the lock on, or a QThread dropped.  These drive the REAL
# methods, not paraphrases of them.
# =========================================================================== #
class _RealRejectFake(FakeAnnotator):
    """Enough surface for the REAL `_reject_split` normal path."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._split_fragment_refine = False
        self._smooth_running = False
        self._interp_snapshot = None
        self._interp_refine_mode = False
        self._isolation_active = False
        self._isolation_label_set = None
        self._refine_baseline = None
        self._split_isolate_was_on = False
        self._split_main_visible = True
        self._navigator_widget = None
        self._detected_peaks = None
        self._auto_detect_active = False
        self._seed_id_base = 0
        self._refine_pre_preview = None
        self._split_fragment_ids = []
        self._fragment_refine_roi = None
        self._reshape_parked_threads = []
        self.selected_all = []
        self.viewer = type("V", (), {"layers": []})()

    def _set_selected_label_all(self, lid):
        self.selected_all.append(lid)

    def _on_isolation_changed(self, s):
        pass

    def _clear_draw_snapshot(self):
        pass

    def _refresh_merge_button_state(self):
        pass

    def _refresh_label_action_buttons_state(self):
        pass

    def _remove_prob_overlay(self):
        pass

    def _reset_smooth_state(self):
        pass


class _LiveThread:
    """A QThread stand-in that reports itself as NOT stopping."""

    def __init__(self):
        self.quit_calls = 0
        self.finished = _StubSignal()

    def quit(self):
        self.quit_calls += 1

    def wait(self, ms):
        return False


class TestRejectDuringTheSolve:
    """Reject is enabled WHILE the worker runs.  It restores the ROI -- but if
    it leaves the worker alive and `_reshape_running` set, session A's result
    lands in session B."""

    def _fake(self):
        labels, _g, _m, _r = two_columns(0.30)
        vol = labels.copy()
        roi = (slice(0, vol.shape[0]), slice(0, vol.shape[1]),
               slice(0, vol.shape[2]))
        snap = vol[roi].copy()
        vol[vol == 7] = 0
        f = _RealRejectFake(vol, roi, 7)
        f.sync_manager.shared_labels["curation_3d_x"] = vol
        f._split_snapshot = snap
        f._refine_layer = FakeLayer(np.zeros_like(vol), "Refine")
        bind(f, "_reject_split", "_restore_reshape_target",
             "_fire_reshape_lock", "_reset_split_state",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_remove_guide_layer", "_teardown_reshape_for_switch",
             "_abandon_reshape_worker")
        return f, vol, snap

    def test_reject_mid_solve_CANCELS_the_worker(self):
        f, _vol, _snap = self._fake()
        cancelled = []

        class W:
            def cancel(self_i):
                cancelled.append(1)

        f._reshape_running = True
        f._reshape_worker = W()
        f._reshape_thread = _LiveThread()
        f._reject_split()
        assert cancelled == [1], (
            "Reject during the solve must cancel the worker, or its answer "
            "lands in the NEXT split session"
        )

    def test_reject_clears_the_running_flag(self):
        f, _vol, _snap = self._fake()
        f._reshape_running = True
        f._reshape_worker = None
        f._reshape_thread = None
        f._reject_split()
        assert f._reshape_running is False

    def test_reject_restores_the_cell(self):
        f, vol, snap = self._fake()
        f._reshape_running = True
        f._reshape_worker = None
        f._reshape_thread = None
        f._reject_split()
        assert np.array_equal(vol, snap)

    def test_reset_split_state_clears_the_running_flag(self):
        f, _vol, _snap = self._fake()
        f._reshape_running = True
        f._reset_split_state()
        assert f._reshape_running is False, (
            "_reset_split_state resets every other split flag; leaving this "
            "one set means the next Preview refuses to start"
        )


class TestStaleGuardComparesTheToken:
    def test_a_result_from_an_EARLIER_session_is_discarded(self):
        """The docstring promises a dispatch token.  Without comparing it, two
        sessions on the same patch/run/layer are indistinguishable."""
        f = FakeAnnotator(np.zeros((4, 4, 4), np.int32), (slice(0, 4),) * 3, 7)
        bind(f, "_reshape_is_stale", "_reshape_context")
        f._reshape_token = 5           # session B is live
        assert f._reshape_is_stale(result_token=4) is True

    def test_the_CURRENT_token_is_not_stale(self):
        f = FakeAnnotator(np.zeros((4, 4, 4), np.int32), (slice(0, 4),) * 3, 7)
        bind(f, "_reshape_is_stale", "_reshape_context")
        f._reshape_token = 5
        assert f._reshape_is_stale(result_token=5) is False


class TestCancelThenPreviewAgain:
    """Cancel re-enables Preview.  If it left the thread in the active slot,
    the next Preview overwrites the only strong reference to a RUNNING QThread
    -- which aborts the process."""

    def _fake(self):
        f = FakeAnnotator(np.zeros((4, 4, 4), np.int32), (slice(0, 4),) * 3, 7)
        f._reshape_parked_threads = []
        bind(f, "_on_reshape_cancel", "_reshape_thread_cleanup",
             "_guide_thread_cleanup", "_abandon_reshape_worker")
        return f

    def test_cancel_parks_a_thread_that_will_not_stop(self):
        f = self._fake()

        class W:
            def cancel(self_i):
                pass

        f._reshape_running = True
        f._reshape_worker = W()
        f._reshape_thread = _LiveThread()
        f._on_reshape_cancel()
        assert f._reshape_thread is None, (
            "the active slot must be empty, or the next Preview overwrites a "
            "running QThread and Qt aborts the process"
        )
        assert len(f._reshape_parked_threads) == 1

    def test_cancel_with_a_thread_that_stops_simply_releases_it(self):
        f = self._fake()

        class W:
            def cancel(self_i):
                pass

        f._reshape_running = True
        f._reshape_worker = W()
        f._reshape_thread = _StubThread(True)
        f._on_reshape_cancel()
        assert f._reshape_thread is None
        assert f._reshape_parked_threads == []


class TestTeardownCleansUpEvenWhenNotRunning:
    def test_a_thread_left_by_a_cancel_is_still_cleaned_on_teardown(self):
        """`_reshape_running` is False after a cancel, so gating the cleanup on
        it would leave the thread for the teardown to drop."""
        f = FakeAnnotator(np.zeros((4, 4, 4), np.int32), (slice(0, 4),) * 3, 7)
        f._reshape_parked_threads = []
        f._split_snapshot = None
        bind(f, "_teardown_reshape_for_switch", "_reshape_thread_cleanup",
             "_guide_thread_cleanup", "_remove_guide_layer",
             "_restore_reshape_target", "_fire_reshape_lock",
             "_abandon_reshape_worker")
        f._reshape_running = False           # cancelled a moment ago
        f._reshape_thread = _StubThread(True)
        f._reshape_worker = _StubWorker()
        f._teardown_reshape_for_switch()
        assert f._reshape_thread is None


class TestAcceptWithNoResultDoesNotStrandTheLock:
    def test_the_bare_return_tears_the_session_down(self):
        import inspect

        src = inspect.getsource(_K()._accept_split)
        i = src.index("_reshape_commit is None")
        head = src[i:i + 1200]
        assert "_teardown_reshape_for_switch()" in head, (
            "accept-with-no-result leaves Accept and Reject disabled, so the "
            "lock must come off here or Step 3 is stuck"
        )

    def test_every_guided_tail_releases_the_lock_in_a_finally(self):
        import inspect

        for name in ("_accept_split", "_reject_split"):
            src = inspect.getsource(getattr(_K(), name))
            assert "finally:" in src, (
                "{} has bare returns while holding the lock".format(name))


class TestDividingVerdictKeepsTheTarget:
    """A 'dividing' verdict is a NO-OP WRITE.  The cell keeps its own
    id and its row stays active -- it is not retired in favour of a fresh id."""

    def _one_piece(self, res):
        import copy as _copy

        class _Arb:
            verdict = "dividing"
            merged_pairs = [(1, 2)]

        one = _copy.copy(res)
        keep = int(res.piece_stats[0]["id"])
        # The arbiter merges the pair back, so the PARTITION comes out as one
        # piece too -- a fixture that kept two would not be a dividing result.
        one.result = np.where(np.asarray(res.result) > 0, keep, 0)
        one.piece_stats = [dict(res.piece_stats[0])]
        one.piece_stats[0]["id"] = keep
        one.arbiter = _Arb()
        return one

    def test_a_single_merged_piece_keeps_the_target_id(self, solved):
        labels, res = solved
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        f = FakeAnnotator(labels.copy(), res.roi, 7, refine=refine)
        f.next_id = 900
        bind(f, "_apply_reshape_result",
         "_print_count_diag_result", "_render_reshape_preview",
             "_reshape_quality_notes", "_reshape_review_status")
        f._apply_reshape_result(self._one_piece(res))
        assert set(f._reshape_final_ids.values()) == {7}, (
            "a kept-whole cell must keep its own id, not be superseded by a "
            "new one"
        )

    def test_the_commit_then_records_dividing_on_the_ORIGINAL_row(self, solved):
        labels, res = solved
        vol = labels.copy()
        vol[vol == 7] = 0
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        f = FakeAnnotator(vol, res.roi, 7, adv=FakeAdv(checked=[7]),
                          refine=refine)
        f.next_id = 900
        bind(f, "_apply_reshape_result",
         "_print_count_diag_result", "_render_reshape_preview",
             "_write_reshape_result", "_reshape_quality_notes",
             "_reshape_review_status")
        f._apply_reshape_result(self._one_piece(res))
        out = f._write_reshape_result(vol, 7)
        assert out["new_ids"] == []
        assert out["superseded"] is False
        assert out["flags"].get(7) == "dividing"

    def test_a_real_split_still_mints_new_ids(self, solved):
        labels, res = solved
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        f = FakeAnnotator(labels.copy(), res.roi, 7, refine=refine)
        f.next_id = 900
        bind(f, "_apply_reshape_result",
         "_print_count_diag_result", "_render_reshape_preview",
             "_reshape_quality_notes", "_reshape_review_status")
        f._apply_reshape_result(res)
        assert 7 not in set(f._reshape_final_ids.values())
        assert min(f._reshape_final_ids.values()) == 900


# =========================================================================== #
# 14. Runtime coverage of the guarded branches (AUDIT 2026-09-09).
#
# These replace source-inspection assertions with calls.  A source check
# passes whether or not the line it greps for does anything; each of these
# fails if the branch is made a no-op.
# =========================================================================== #
class TestGuardedBranchesAtRuntime:
    def test_BOTH_of_todays_seed_sources_stay_on_the_legacy_path(self, qtless=None):
        """The regression lock covers `painted` AND `detected` (AUDIT
        2026-09-09: `detected` used to route into the engine, which made its
        "(today)" label a lie and left half the legacy surface unlocked)."""
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        assert app is not None
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget, SEEDS_FROM_DETECTED, SEEDS_FROM_PAINTED,
            WALLS_FROM_MEMBRANE, WALLS_FROM_SHAPE,
        )

        w = AdvancedFeaturesWidget(mode="3d_curation")
        try:
            for seeds in (SEEDS_FROM_PAINTED, SEEDS_FROM_DETECTED):
                w.split_seeds_from_combo.setCurrentIndex(seeds)
                w.split_walls_from_combo.setCurrentIndex(WALLS_FROM_SHAPE)
                assert w.is_guided_split() is False, (
                    "seed source {} is one of today's two; it must keep its "
                    "existing code path".format(seeds)
                )
                # ... but asking for a membrane wall DOES need the engine
                w.split_walls_from_combo.setCurrentIndex(WALLS_FROM_MEMBRANE)
                assert w.is_guided_split() is True
        finally:
            w.setParent(None)

    def test_the_manual_path_no_longer_reads_a_channel_at_all(self):
        """The live driver's d13, retired with the branch that caused it.

        With the membrane METHOD chosen and the legacy image combo blank,
        `_preview_split` used to fall back to the first image layer it
        could find -- a wrong-channel split with no message -- and then
        got an explicit refusal instead.  Brick 2b removed the branch:
        `is_guided_split()` routes every wall index past "shape only"
        into the engine, so the manual path is shape-only and never
        touches an image.  The engine's own refusal (pinned in
        TestRecipe) is the one that survives.

        Asserted the way a REMOVAL has to be: the source of the method
        carries no channel read, and a preview with no image layers
        anywhere still runs.
        """
        import inspect

        src = inspect.getsource(_K()._preview_split)
        # CODE only: the comment that records the removal names the two
        # combos, and a test that could not tell a comment from a read
        # would fail on its own explanation.
        code = "".join(
            line for line in src.splitlines(True)
            if not line.lstrip().startswith("#")
        )
        assert "split_image_combo" not in code
        assert "split_method_combo" not in code
        assert "intensity" not in code.split("valid, seed_ids")[0]

    def test_preview_routes_to_the_engine_ONLY_when_guided(self):
        """Both directions, by calling the real `_preview_split`.

        The predicate is what the PICKERS say, read live -- not a flag
        latched at Start (2026-09-18): the user chooses the source after
        Start far more often than before it.
        """
        M = _M()
        cases = (
            ({"walls_from": M.WALLS_FROM_SHAPE}, []),
            ({"walls_from": M.WALLS_FROM_MEMBRANE}, ["guided"]),
            ({"seeds_from": M.SEEDS_FROM_IMAGE}, ["guided"]),
        )
        for params, expected in cases:
            f = FakeAnnotator(np.zeros((4, 4, 4), np.int32),
                              (slice(0, 2),) * 3, 7, adv=FakeAdv(params))
            f._split_running = True
            f._refine_baseline = np.zeros((2, 2, 2), np.int32)
            f._refine_layer = None
            seen = []
            f._preview_reshape = lambda: seen.append("guided")
            bind(f, "_preview_split", "_fire_reshape_lock")
            f._refresh_reshape_inputs = lambda: None
            f._preview_split()
            assert seen == expected, params

    def test_accept_with_no_result_RESTORES_and_would_fail_if_it_did_not(self):
        """Drives the real restore: the volume must come back voxel for
        voxel, so stubbing `_restore_reshape_target` out fails this."""
        labels, _g, _m, _r = two_columns(0.30)
        vol = labels.copy()
        roi = (slice(0, vol.shape[0]), slice(0, vol.shape[1]),
               slice(0, vol.shape[2]))
        snap = vol[roi].copy()
        vol[vol == 7] = 0
        f = FakeAnnotator(vol, roi, 7)
        f._split_snapshot = snap
        bind(f, "_restore_reshape_target")
        assert 7 not in set(np.unique(vol))     # the hole is real
        f._restore_reshape_target("test")
        assert np.array_equal(vol, snap)
        assert 7 in set(np.unique(vol))

    def test_reject_really_restores_via_the_real_method(self):
        labels, _g, _m, _r = two_columns(0.30)
        vol = labels.copy()
        roi = (slice(0, vol.shape[0]), slice(0, vol.shape[1]),
               slice(0, vol.shape[2]))
        snap = vol[roi].copy()
        vol[vol == 7] = 0
        f = _RealRejectFake(vol, roi, 7)
        f.sync_manager.shared_labels["curation_3d_x"] = vol
        f._split_snapshot = snap
        f._refine_layer = FakeLayer(np.zeros_like(vol), "Refine")
        f._reshape_lock_fired = True
        bind(f, "_reject_split", "_restore_reshape_target",
             "_fire_reshape_lock", "_reset_split_state",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_remove_guide_layer", "_teardown_reshape_for_switch",
             "_abandon_reshape_worker")
        f._reject_split()
        assert np.array_equal(vol, snap)
        assert f.lock_calls == [False], "Reject must release the lock"
        assert f._split_running is False

    def test_the_lock_comes_off_even_when_the_reject_body_raises(self):
        f = _RealRejectFake(np.zeros((4, 4, 4), np.int32),
                            (slice(0, 4),) * 3, 7)
        f._reshape_lock_fired = True
        f._split_fragment_refine = False

        class _Boom(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom")

        f.sync_manager.shared_labels = _Boom()
        bind(f, "_reject_split", "_restore_reshape_target",
             "_fire_reshape_lock", "_reset_split_state",
             "_reshape_thread_cleanup", "_guide_thread_cleanup",
             "_remove_guide_layer", "_teardown_reshape_for_switch",
             "_abandon_reshape_worker")
        with pytest.raises(RuntimeError):
            f._reject_split()
        assert f.lock_calls == [False], (
            "a raise inside Reject must not strand the Step 3 lock")


# =========================================================================== #
# 15. The guide layer de-registers from the SyncManager (AUDIT 2026-09-09).
# =========================================================================== #
class TestGuideLayerDeregisters:
    def test_remove_disconnects_the_label_sync_handlers(self):
        f = FakeAnnotator(np.zeros((2, 2, 2), np.int32), (slice(0, 2),) * 3, 7)
        layer = FakeLayer(np.zeros((2, 2, 2)), "guide_x_p")
        f._reshape_guide_layer = layer
        f._reshape_guide = np.zeros((2, 2, 2))
        seen = []
        f.sync_manager.disconnect_label_sync = lambda l: seen.append(l)
        f.viewer = type("V", (), {"layers": []})()
        bind(f, "_remove_guide_layer")
        f._remove_guide_layer()
        assert seen == [layer], (
            "connect_label_mode_sync registers ANY Labels layer that becomes "
            "active; nothing cleans up a temp one, so the handlers would keep "
            "the removed layer and its full-volume array alive"
        )
        assert f._reshape_guide_layer is None
        assert f._reshape_guide is None

    def test_remove_still_hides_before_it_removes(self):
        f = FakeAnnotator(np.zeros((2, 2, 2), np.int32), (slice(0, 2),) * 3, 7)
        layer = FakeLayer(np.zeros((2, 2, 2)), "guide_x_p")
        order = []

        class _Layers(list):
            def remove(self, l):
                order.append(("remove", bool(l.visible)))
                list.remove(self, l)

        layers = _Layers([layer])
        f.viewer = type("V", (), {"layers": layers})()
        f._reshape_guide_layer = layer
        f.sync_manager.disconnect_label_sync = lambda l: None
        bind(f, "_remove_guide_layer")
        f._remove_guide_layer()
        assert order == [("remove", False)], "hide before remove"


# =========================================================================== #
# 16. Brick 2a -- the recipe is TOTAL, and it names two channels
# =========================================================================== #
class _ImgLayer(FakeLayer):
    """A stand-in that `isinstance(..., Image)` accepts."""


def _image_fake(**params):
    """A fake whose viewer resolves two named image layers."""
    from napari.layers import Image

    M = _M()
    vol = np.zeros((6, 6, 6), np.int32)
    vol[2:4, 2:4, 2:4] = 7
    roi = (slice(0, 6),) * 3
    refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
    base = dict(seeds_from=M.SEEDS_FROM_IMAGE, walls_from=M.WALLS_FROM_MEMBRANE,
                seed_detector=0, seed_channel="nuclei", wall_channel="membrane",
                seed_preproc="clahe", wall_preproc="none", search_padding=15)
    base.update(params)
    fake = FakeAnnotator(vol, roi, 7, adv=FakeAdv(base), refine=refine)

    layers = {}
    for name in ("nuclei", "membrane"):
        layer = Image(np.zeros((6, 6, 6), np.float32), name=name)
        layers[name] = layer
    fake.viewer = type("V", (), {"layers": list(layers.values())})()
    fake._layer_by_name = lambda _layers, name: layers.get(name)
    fake._squeeze_image_channel = lambda a: a
    bind(fake, "_reshape_recipe")
    return fake, M


class TestTheRecipeMappingIsTotal:
    def test_every_seed_source_index_maps_or_refuses(self):
        """Rule 5: a combo one item ahead of this table is a
        KeyError, one behind is a silently different algorithm.  So every
        index the combo can produce must land on a validated token."""
        from packages.epicycle_annotator.reshape.spec import SEEDS

        M = _M()
        for idx in range(len(M.SEEDS_FROM_LABELS)):
            fake, _ = _image_fake(seeds_from=idx, walls_from=1)
            fake._reshape_guide = np.zeros((6, 6, 6), np.int32)
            fake._split_seed_ids = [7, 101]
            fake._refine_layer.data[0, 0, 0] = 101
            spec, images, _seeds = fake._reshape_recipe()
            assert spec is not None, (idx, images)
            assert spec.seeds in SEEDS, (idx, spec.seeds)

    def test_every_detector_index_maps_to_an_image_seed_token(self):
        from packages.epicycle_annotator.reshape.spec import IMAGE_DERIVED_SEEDS

        M = _M()
        for det in range(len(M.SEED_DETECTORS)):
            fake, _ = _image_fake(seed_detector=det)
            spec, _images, _seeds = fake._reshape_recipe()
            assert spec is not None, det
            assert spec.seeds == M.SEED_DETECTORS[det]
            assert spec.seeds in IMAGE_DERIVED_SEEDS

    def test_every_wall_index_maps_to_a_validated_token(self):
        from packages.epicycle_annotator.reshape.spec import WALLS

        M = _M()
        for idx in range(len(M.WALL_KINDS)):
            fake, _ = _image_fake(walls_from=idx)
            spec, images, _seeds = fake._reshape_recipe()
            assert spec is not None, (idx, images)
            assert spec.walls == M.WALL_KINDS[idx]
            assert spec.walls in WALLS

    def test_a_seed_index_off_the_end_REFUSES_rather_than_falling_back(self):
        fake, _ = _image_fake(seeds_from=99)
        spec, reason, _ = fake._reshape_recipe()
        assert spec is None
        assert "99" in str(reason)

    def test_a_wall_index_off_the_end_REFUSES(self):
        fake, _ = _image_fake(walls_from=42)
        spec, reason, _ = fake._reshape_recipe()
        assert spec is None
        assert "42" in str(reason)

    def test_a_detector_index_off_the_end_REFUSES(self):
        fake, _ = _image_fake(seed_detector=17)
        spec, reason, _ = fake._reshape_recipe()
        assert spec is None
        assert "17" in str(reason)


class TestTheRecipeCarriesBothChannels:
    def test_images_holds_the_membrane_AND_the_nuclei_arrays(self):
        fake, _ = _image_fake()
        spec, images, _seeds = fake._reshape_recipe()
        assert set(images) == {"membrane", "nuclei"}
        assert spec.params["wall_channel"] == "membrane"
        assert spec.params["seed_channel"] == "nuclei"

    def test_a_geometry_wall_asks_for_no_membrane_image(self):
        M = _M()
        fake, _ = _image_fake(walls_from=M.WALL_KINDS.index("skeleton"))
        spec, images, _seeds = fake._reshape_recipe()
        assert set(images) == {"nuclei"}
        assert spec.params["wall_channel"] is None

    def test_a_non_image_marker_source_asks_for_no_nuclei_image(self):
        M = _M()
        fake, _ = _image_fake(seeds_from=M.SEEDS_FROM_RUN_LABELS)
        fake._reshape_guide = np.zeros((6, 6, 6), np.int32)
        spec, images, _seeds = fake._reshape_recipe()
        assert set(images) == {"membrane"}
        assert spec.params["seed_channel"] is None

    def test_a_blank_marker_channel_REFUSES_instead_of_grabbing_one(self):
        """inv. 3: five combos, one policy -- a blank one is an explicit
        error, never the first image layer."""
        fake, _ = _image_fake(seed_channel="")
        spec, reason, _ = fake._reshape_recipe()
        assert spec is None
        assert "channel" in str(reason).lower()

    def test_the_preparation_chains_reach_the_spec(self):
        fake, _ = _image_fake(seed_preproc="zcorrect+clahe",
                              wall_preproc="denoise")
        spec, _images, _seeds = fake._reshape_recipe()
        assert spec.params["seed_preproc"] == "zcorrect+clahe"
        assert spec.params["wall_preproc"] == "denoise"

    def test_the_arbiter_survives_EVERY_image_derived_wall(self):
        from packages.epicycle_annotator.reshape.spec import IMAGE_DERIVED_WALLS

        M = _M()
        for idx, kind in enumerate(M.WALL_KINDS):
            fake, _ = _image_fake(walls_from=idx, arbiter=True)
            spec, _images, _seeds = fake._reshape_recipe()
            assert spec.arbiter is (kind in IMAGE_DERIVED_WALLS), kind


# =========================================================================== #
# 17. Brick 2a -- the temporary-layer registry
# =========================================================================== #
class TestTheTempLayerRegistry:
    def _fake(self):
        f = FakeAnnotator(np.zeros((2, 2, 2), np.int32), (slice(0, 2),) * 3, 7)

        class _Layers(list):
            def remove(self, layer):
                list.remove(self, layer)

        added = _Layers()

        def _add_labels(array, name=None, opacity=None):
            layer = FakeLayer(array, name)
            added.append(layer)
            return layer

        f.viewer = type("V", (), {
            "layers": added, "add_labels": staticmethod(_add_labels)})()
        f.viewer.layers = added
        f.viewer.add_labels = _add_labels
        bind(f)
        return f, added

    def test_the_mask_and_the_markers_coexist(self):
        """The single slot this replaces would have made the markers EVICT
        the loaded mask -- they have different lifetimes."""
        f, added = self._fake()
        f._add_temp_layer("mask", np.zeros((2, 2, 2), np.int32))
        f._add_temp_layer("markers", np.zeros((2, 2, 2), np.int32))
        assert len(added) == 2
        assert set(f._temp_guide_layers) == {"mask", "markers"}

    def test_the_mask_keeps_its_legacy_slot(self):
        f, _added = self._fake()
        layer = f._add_temp_layer("mask", np.zeros((2, 2, 2), np.int32))
        assert f._reshape_guide_layer is layer
        f._remove_temp_layer("mask")
        assert f._reshape_guide_layer is None

    def test_removing_the_markers_leaves_the_mask_alone(self):
        f, added = self._fake()
        f._add_temp_layer("mask", np.zeros((2, 2, 2), np.int32))
        f._add_temp_layer("markers", np.zeros((2, 2, 2), np.int32))
        f._remove_temp_layer("markers")
        assert set(f._temp_guide_layers) == {"mask"}
        assert len(added) == 1

    def test_every_temp_layer_is_unmanaged_read_only_and_brush_locked(self):
        f, _added = self._fake()
        layer = f._add_temp_layer("markers", np.zeros((2, 2, 2), np.int32))
        assert layer._epicycle_temporary is True
        assert layer.editable is False
        assert layer.mode == "pan_zoom"
        assert getattr(layer, "_epicycle_brush_locked", False) is True

    def test_the_default_name_carries_the_sweep_prefix(self):
        """The controller's _clear_curation_layers sweep keys on it."""
        f, _added = self._fake()
        layer = f._add_temp_layer("markers", np.zeros((2, 2, 2), np.int32))
        assert layer.name == "guide_markers"

    def test_adding_the_same_key_twice_replaces_rather_than_leaks(self):
        f, added = self._fake()
        f._add_temp_layer("markers", np.zeros((2, 2, 2), np.int32))
        f._add_temp_layer("markers", np.zeros((2, 2, 2), np.int32))
        assert len(added) == 1

    def test_removing_a_key_that_was_never_added_is_safe(self):
        f, _added = self._fake()
        f._remove_temp_layer("markers")        # must not raise

    def test_remove_all_clears_both(self):
        f, added = self._fake()
        f._add_temp_layer("mask", np.zeros((2, 2, 2), np.int32))
        f._add_temp_layer("markers", np.zeros((2, 2, 2), np.int32))
        f._remove_all_temp_layers()
        assert f._temp_guide_layers == {}
        assert added == []


class TestTheMarkersAreRemovedAtEveryExit:
    """Five sites: Accept, Discard, Stop, patch switch, close.

    U1b retired `guide_markers` -- no found source draws a second layer
    any more -- but the five sweeps stay: the registry is shared with
    Brick U3's `guide_walls` and with the loaded mask, and a sweep that
    finds nothing is the cheapest kind of correctness there is.
    """

    MARKER_SITES = (
        "_accept_split", "_reject_split", "_on_reshape_cancel",
        "_teardown_reshape_for_switch", "_close_cleanup",
    )

    @pytest.mark.parametrize("method", MARKER_SITES)
    def test_the_site_removes_the_markers(self, method):
        import inspect

        src = inspect.getsource(getattr(_K(), method))
        # The teardown sweeps the WHOLE registry (there is no reason for a
        # patch switch to leave any temporary layer behind); the other four
        # name the markers key.
        removes = (
            '_remove_temp_layer("markers")' in src
            or "_remove_all_temp_layers()" in src
        )
        assert removes, (
            "{} leaves guide_markers on the volume".format(method)
        )

    def test_the_MASK_is_removed_at_exactly_its_own_two_sites(self):
        """It deliberately SURVIVES Accept/Discard so the next cell reuses
        it, so the count is two, not five."""
        import inspect

        sites = []
        for name in ("_accept_split", "_reject_split", "_on_reshape_cancel",
                     "_teardown_reshape_for_switch", "_close_cleanup",
                     "_on_guide_loaded"):
            src = inspect.getsource(getattr(_K(), name))
            if "_remove_guide_layer()" in src:
                sites.append(name)
        assert sorted(sites) == ["_on_guide_loaded",
                                 "_teardown_reshape_for_switch"]

    def test_the_marker_worker_is_abandoned_on_teardown_and_close(self):
        import inspect

        for name in ("_teardown_reshape_for_switch", "_close_cleanup"):
            src = inspect.getsource(getattr(_K(), name))
            assert "_abandon_seed_find_worker()" in src, name


# =========================================================================== #
# 18. Brick 2a -- ONE Stop button, two jobs
# =========================================================================== #
class _FindFake(FakeAnnotator):
    def __init__(self):
        super().__init__(np.zeros((2, 2, 2), np.int32), (slice(0, 2),) * 3, 7)
        self.viewer = type("V", (), {"layers": []})()


class TestTheStopButtonRoutes:
    def _fake(self, kind):
        f = _FindFake()
        f._split_busy_kind = kind
        f.abandoned = []
        bind(f, "_on_reshape_cancel")
        # AFTER bind(): the shared binder also installs the real
        # `_abandon_seed_find_worker`, which would shadow the recorder.
        f._abandon_reshape_worker = lambda: f.abandoned.append("solve")
        f._abandon_seed_find_worker = lambda: f.abandoned.append("find")
        f._update_split_ui = lambda **kw: f.ui_calls.append(kw)
        return f

    def test_stopping_a_FIND_does_not_tear_down_the_solve(self):
        f = self._fake("find")
        f._on_reshape_cancel()
        assert f.abandoned == ["find"]

    def test_stopping_a_SOLVE_does_not_tear_down_the_find(self):
        f = self._fake("solve")
        f._on_reshape_cancel()
        assert f.abandoned == ["solve"]

    def test_with_nothing_running_it_falls_through_to_the_solve_path(self):
        f = self._fake(None)
        f._on_reshape_cancel()
        assert f.abandoned == ["solve"]

    def test_the_find_path_says_the_cell_is_still_held_out(self):
        f = self._fake("find")
        f._on_reshape_cancel()
        texts = " ".join(str(c.get("status_text", "")) for c in f.ui_calls)
        assert "held out" in texts

    def test_a_find_launch_sets_the_kind_and_a_solve_launch_sets_the_other(self):
        import inspect

        find_src = inspect.getsource(_K()._find_reshape_markers)
        solve_src = inspect.getsource(_K()._preview_reshape)
        assert '_split_busy_kind = "find"' in find_src
        assert '_split_busy_kind = "solve"' in solve_src


# =========================================================================== #
# 19. Brick 2a -- the Find sees exactly what the Preview will see
# =========================================================================== #
class TestTheFindAndThePreviewAgree:
    def test_both_build_the_labels_array_through_ONE_helper(self):
        """Two different arrays would mean two different crops, two
        different sets of markers and two different sets of ids -- so
        the ticks would select the wrong markers.  Asserted
        as IDENTITY: both launches must hand the worker the very array
        the one helper produced."""
        f, seen = _launch_recorder()
        f._find_reshape_markers()
        f._preview_reshape()
        assert len(seen) == 2, seen
        assert seen[0]["labels"] is f.the_labels
        assert seen[1]["labels"] is f.the_labels

    def test_the_helper_puts_the_PRISTINE_target_in_the_roi(self):
        f = FakeAnnotator(np.zeros((6, 6, 6), np.int32), (slice(1, 3),) * 3, 7)
        main = FakeLayer(np.zeros((6, 6, 6), np.int32), "curation_3d_x")
        f._layer_by_name = lambda _layers, _name: main
        f.viewer = type("V", (), {"layers": []})()
        f._refine_baseline = np.full((2, 2, 2), 7, np.int32)
        bind(f, "_reshape_labels_for_engine")
        out = f._reshape_labels_for_engine()
        assert out.shape == (6, 6, 6)
        assert (out[1:3, 1:3, 1:3] == 7).all()
        assert out.sum() == 7 * 8, "nothing outside the ROI was written"

    def test_the_helper_answers_None_when_there_is_no_session(self):
        f = FakeAnnotator(np.zeros((2, 2, 2), np.int32), None, 7)
        f._layer_by_name = lambda _layers, _name: None
        f.viewer = type("V", (), {"layers": []})()
        bind(f, "_reshape_labels_for_engine")
        assert f._reshape_labels_for_engine() is None

    def test_both_pass_the_SAME_roi(self):
        """The detectors normalise over the CROP and number by
        enumeration order, so two crops give two different sets of
        ids."""
        f, seen = _launch_recorder()
        f._find_reshape_markers()
        f._preview_reshape()
        assert seen[0]["roi"] == f._split_roi
        assert seen[1]["roi"] == f._split_roi


class TestTheCherryPickReachesTheSolve:
    def test_an_empty_pick_REFUSES_rather_than_cutting_blind(self):
        import inspect

        src = inspect.getsource(_K()._preview_reshape)
        assert "get_checked_marker_ids" in src
        assert "unticked every marker" in src
        assert "keep_seed_ids=keep_seed_ids" in src

    def test_None_means_find_them_fresh_and_keep_them_all(self):
        """No Find happened, so the engine finds them itself -- the
        panel must not turn that into an empty keep-list."""
        f, seen = _launch_recorder()
        f._adv.marker_ids = None
        f._preview_reshape()
        assert seen[0]["keep_seed_ids"] is None

    def test_a_cherry_pick_reaches_the_worker_verbatim(self):
        f, seen = _launch_recorder()
        f._adv.marker_ids = [2, 5]
        f._preview_reshape()
        assert seen[0]["keep_seed_ids"] == [2, 5]

    def test_an_EMPTY_pick_never_reaches_the_worker_at_all(self):
        f, seen = _launch_recorder()
        f._adv.marker_ids = []
        f._preview_reshape()
        assert seen == [], "it cut with no markers"
        texts = " ".join(str(c.get("marker_note", ""))
                         for c in f._adv.ui_calls)
        assert "unticked every marker" in texts


# =========================================================================== #
# 20. Brick 2a -- Search padding is read at Start
# =========================================================================== #
class TestSearchPaddingIsReadAtStart:
    def test_the_snapshot_helper_takes_a_margin(self):
        import inspect

        sig = inspect.signature(_K()._snapshot_label_roi)
        assert "margin" in sig.parameters
        assert sig.parameters["margin"].default == 15, (
            "the default must stay today's behaviour"
        )

    def test_the_margin_actually_changes_the_roi(self):
        vol = np.zeros((40, 40, 40), np.int32)
        vol[20, 20, 20] = 7
        roi_default, _ = _K()._snapshot_label_roi(vol, 7)
        roi_small, _ = _K()._snapshot_label_roi(vol, 7, margin=2)
        assert roi_default[0].start == 5
        assert roi_small[0].start == 18

    def test_a_negative_margin_is_clamped_not_inverted(self):
        vol = np.zeros((40, 40, 40), np.int32)
        vol[20, 20, 20] = 7
        roi, _ = _K()._snapshot_label_roi(vol, 7, margin=-5)
        assert roi[0].start == 20 and roi[0].stop == 21

    def test_start_reads_the_knob_into_the_snapshot(self):
        import inspect

        src = inspect.getsource(_K()._start_split)
        assert "search_padding" in src
        i_read = src.index("search_padding")
        i_snap = src.index("_snapshot_label_roi(")
        assert i_read < i_snap, "the knob must be read BEFORE the snapshot"
        assert "margin=_padding" in src


# =========================================================================== #
# 21. Fix pass -- the stale token has to come from the FINISHING worker
# =========================================================================== #
class _TokenWorker:
    """A stand-in for a worker that carries the token it was launched with."""

    def __init__(self, token):
        self._epicycle_token = token


class _HandlerFake(FakeAnnotator):
    """A fake whose `sender()` seam can be pointed at any worker."""

    def __init__(self):
        super().__init__(np.zeros((4, 4, 4), np.int32), (slice(0, 2),) * 3, 7)
        self.viewer = type("V", (), {"layers": []})()
        self._sender = None
        self.removed_keys = []
        self.cleanups = 0
        self._refine_baseline = np.zeros((2, 2, 2), np.int32)

    def _signal_sender(self):
        return self._sender

    def _remove_temp_layer(self, key):
        self.removed_keys.append(key)

    def _seed_find_thread_cleanup(self):
        self.cleanups += 1

    def _reshape_thread_cleanup(self):
        self.cleanups += 1

    # Brick U1b: sources 1-4 materialise their markers into the Refine
    # layer.  Source 0 does NOT -- its markers are already on that
    # layer -- and a fake that answers True unconditionally can never
    # take the other branch (audit M4).  So this reads the SOURCE, the
    # way production does, and `_source` below picks it.
    _source = 4

    def _marker_source_materialises(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            MATERIALISED_MARKER_SOURCES,
        )

        return int(self._source) in MATERIALISED_MARKER_SOURCES

    def _materialise_found_markers(self, report, markers):
        """Stands in for the real write, but does NOT invent its verdict.

        `_materialise_ok` used to be set here, which meant the handler's
        two branches were chosen by the fake rather than exercised.  The
        flag is now whatever the REAL method's contract says it is:
        non-empty ids -> True.  `_materialised_ids` is filled too, since
        that is the recipe's receipt (audit H1) and a fake that leaves it
        empty would quietly disagree with the branch it just took.
        """
        self.removed_keys.append("materialised")
        ids = [int(m["id"]) for m in markers]
        self._materialise_ok = bool(ids)
        self._materialised_ids = list(ids)
        return (
            ids,
            {int(m["id"]): {"voxels": int(m.get("voxels", 0)),
                            "centroid_zyx": m.get("centroid_zyx")}
             for m in markers},
        )

    def _restore_ready_to_draw_state(self):
        pass

    def _apply_reshape_result(self, result):
        self.removed_keys.append("applied")


class TestSourceZeroDoesNotMaterialise:
    """The branch the fake used to make unreachable (audit M4).

    Hand-painted markers are already ON the Refine layer, so a Find for
    source 0 has nothing to write there.  A fake that answered "yes,
    materialise" for every source meant this half of
    `_on_seed_find_finished` was never run by any test -- and it is the
    half that keeps the seed crop, which the recipe then needs.
    """

    def _fake(self, source):
        f = _HandlerFake()
        f._source = source
        f._seed_find_token = 5
        f._seed_find_launch_token = 5
        f._seed_find_running = True
        f._sender = _TokenWorker(5)
        bind(f, "_seed_find_is_stale", "_on_seed_find_finished")
        return f

    def test_source_0_never_calls_the_materialise(self):
        f = self._fake(0)
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        assert "materialised" not in f.removed_keys, f.removed_keys

    def test_source_0_KEEPS_the_seed_crop(self):
        """It is the only marker set there is on that path."""
        f = self._fake(0)
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        assert f._reshape_found_seeds is not None

    def test_source_0_records_nothing_as_materialised(self):
        f = self._fake(0)
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        assert not getattr(f, "_materialised_ids", [])

    @pytest.mark.parametrize("source", [1, 2, 3, 4])
    def test_every_found_source_DOES_materialise(self, source):
        f = self._fake(source)
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        assert "materialised" in f.removed_keys, source
        assert f._materialise_ok is True
        assert f._materialised_ids == [1, 2]

    @pytest.mark.parametrize("source", [1, 2, 3, 4])
    def test_a_materialised_source_drops_the_crop(self, source):
        f = self._fake(source)
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        assert f._reshape_found_seeds is None


class _Report:
    def __init__(self, status="ok", markers=(), messages=()):
        self.status = status
        self.markers = [
            {"id": i, "voxels": 100 * i,
             "centroid_zyx": (float(i), 2.0 * i, 3.0 * i)}
            for i in markers
        ]
        self.messages = list(messages)
        self.n_found = len(self.markers)
        self.n_in_region = len(self.markers)
        self.roi = (slice(0, 2),) * 3
        self.seeds = np.zeros((2, 2, 2), np.int32)


class TestTheStaleTokenComesFromTheWorker:
    """The token leg was a TAUTOLOGY: the handler read
    `self._seed_find_launch_token`, which the launch sets to the same value
    as `self._seed_find_token`, so the one guard the other three cannot
    replace (Find, Discard, Find again on the same patch/run/layer) never
    fired."""

    def _fake(self, live_token=2, result_token=1):
        f = _HandlerFake()
        f._seed_find_token = live_token
        f._seed_find_launch_token = live_token
        f._seed_find_running = True
        f._sender = _TokenWorker(result_token)
        bind(f, "_seed_find_is_stale", "_on_seed_find_finished",
             "_on_seed_find_error")
        # AFTER bind(): the shared binder installs the REAL `_signal_sender`
        # (which calls Qt's sender()), so the seam is re-pointed here.
        f._signal_sender = lambda: f._sender
        f._remove_temp_layer = lambda key: f.removed_keys.append(key)
        f._seed_find_thread_cleanup = lambda: setattr(
            f, "cleanups", f.cleanups + 1)
        return f

    def test_a_result_from_an_EARLIER_find_is_discarded(self):
        f = self._fake(live_token=2, result_token=1)
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        assert f._adv.ui_calls == [], "a stale find touched the panel"
        assert "materialised" not in f.removed_keys

    def test_a_stale_result_does_NOT_tear_down_the_LIVE_worker(self):
        """The handler used to clean the thread slot BEFORE the gate, so a
        late answer detached the worker session B was waiting on."""
        f = self._fake(live_token=2, result_token=1)
        f._on_seed_find_finished(_Report(markers=(1,)))
        assert f.cleanups == 0
        assert f._seed_find_running is True, (
            "a stale result cleared the flag the LIVE find owns"
        )

    def test_the_CURRENT_token_is_not_stale(self):
        f = self._fake(live_token=2, result_token=2)
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        assert f.cleanups == 1
        assert f._seed_find_running is False
        assert any("marker_ids" in c for c in f._adv.ui_calls)

    def test_an_unknown_sender_falls_back_to_the_other_three_guards(self):
        """Outside Qt there is no sender; the patch / run / layer guards
        still apply and the token leg simply abstains."""
        f = self._fake(live_token=2, result_token=2)
        f._sender = None
        f._on_seed_find_finished(_Report(markers=(1,)))
        assert f.cleanups == 1

    def test_the_error_handler_uses_the_same_gate(self):
        f = self._fake(live_token=2, result_token=1)
        f._on_seed_find_error("boom")
        assert f._adv.ui_calls == []
        assert f.cleanups == 0
        assert f._seed_find_running is True


class TestTheSolvesTokenComesFromTheWorkerToo:
    """`_on_reshape_finished` had the IDENTICAL pre-existing tautology."""

    def _fake(self, live_token=2, result_token=1):
        f = _HandlerFake()
        f._reshape_token = live_token
        f._reshape_launch_token = live_token
        f._reshape_running = True
        f._sender = _TokenWorker(result_token)
        bind(f, "_reshape_is_stale", "_on_reshape_finished")
        f._signal_sender = lambda: f._sender
        f._reshape_thread_cleanup = lambda: setattr(
            f, "cleanups", f.cleanups + 1)
        f._apply_reshape_result = lambda r: f.removed_keys.append("applied")
        return f

    def test_a_result_from_an_EARLIER_preview_is_discarded(self):
        f = self._fake(live_token=2, result_token=1)
        f._on_reshape_finished(_Report(markers=(1, 2)))
        assert "applied" not in f.removed_keys

    def test_a_stale_solve_does_NOT_tear_down_the_live_worker(self):
        f = self._fake(live_token=2, result_token=1)
        f._on_reshape_finished(_Report(markers=(1,)))
        assert f.cleanups == 0
        assert f._reshape_running is True

    def test_the_current_solve_token_is_not_stale(self):
        f = self._fake(live_token=2, result_token=2)
        f._on_reshape_finished(_Report(markers=(1, 2)))
        assert "applied" in f.removed_keys

    def test_both_launches_stamp_the_token_on_the_worker(self):
        import inspect

        for name in ("_find_reshape_markers", "_preview_reshape"):
            src = inspect.getsource(getattr(_K(), name))
            assert "_epicycle_token" in src, (
                "{} does not stamp the launch token on its worker".format(name)
            )


class TestALateFindDoesNotStealTheSolvesBusyBar:
    """`_set_marker_busy(False)` toggles the bar and the Stop button, which
    the SOLVE also owns -- and the handlers released it BEFORE the stale
    gate, so a late Find hid a running solve's Stop button."""

    def test_the_release_happens_after_the_gate(self):
        import inspect

        for name in ("_on_seed_find_finished", "_on_seed_find_error"):
            src = inspect.getsource(getattr(_K(), name))
            i_gate = src.index("_seed_find_is_stale")
            i_release = src.index("marker_busy=False")
            assert i_gate < i_release, (
                "{} releases the busy state before it knows the result is "
                "still wanted".format(name)
            )


class TestPreviewRefusesWhileAFindIsRunning:
    def test_a_preview_during_a_find_is_refused_with_a_reason(self):
        f = _HandlerFake()
        f._seed_find_running = True
        f._reshape_running = False
        bind(f, "_preview_reshape")
        f._preview_reshape()
        texts = " ".join(str(c.get("status_text", "")) for c in f.ui_calls)
        assert "marker" in texts.lower(), texts
        assert f._reshape_running is False, "the solve must not have started"


class TestEveryPathThatEmptiesTheListReconcilesParking:
    """Audit H2.  The escape hatch was wired at ONE site.

    Five other paths call `update_split_ui_state(marker_ids=[])`, and
    `_populate_marker_list` restyles under `blockSignals` and never
    fires the check callback -- so the list emptied while the stash kept
    the voxels.  The next Add marker repopulates with `preserve=True`
    over an EMPTY list, so the row comes back TICKED while its voxels
    are still stashed: S1 says "3 ticked" and the engine is handed 2.

    Each test drives the REAL handler and asks the one question that
    matters: do the list and the layer agree afterwards?
    """

    def _parked(self, source=4):
        """A session with marker 101 painted, unticked and parked."""
        roi = (slice(0, 4),) * 3
        refine = FakeLayer(np.zeros((4, 4, 4), np.int32), "Refine")
        refine.data[0:2, 0:2, 0:2] = 7          # the object
        refine.data[0, 0, 0] = 101              # the marker, on top of it
        f = _HandlerFake()
        f._source = source
        f._split_roi = roi
        f._refine_layer = refine
        f._refine_baseline = np.zeros((4, 4, 4), np.int32)
        f._refine_baseline[0:2, 0:2, 0:2] = 7
        f._split_label_id = 7
        f._split_painted_marker_ids = [101]
        f._adv.set_marker_rows([101], off=[101])
        bind(f)                      # `_ALWAYS_BOUND`, parking included
        f._sync_marker_parking()
        assert f._parked_marker_ids() == [101], "setup failed to park"
        assert int((refine.data == 101).sum()) == 0
        return f

    def _assert_agrees(self, f):
        on_layer = int((f._refine_layer.data == 101).sum())
        rows = f._adv._marker_rows
        assert not f._parked_marker_ids(), (
            "the stash outlived the row that could have un-parked it")
        assert on_layer > 0, "the marker is nowhere: not on the layer, "
        assert 101 not in rows or 101 not in f._adv._marker_off

    def test_the_find_launch_path(self):
        f = self._parked()
        f._adv.update_split_ui_state(marker_ids=[], marker_busy=True)
        f._sync_marker_parking()
        self._assert_agrees(f)

    def test_the_seed_find_error_path(self):
        f = self._parked()
        f._seed_find_token = 5
        f._seed_find_launch_token = 5
        f._seed_find_running = True
        f._sender = _TokenWorker(5)
        bind(f, "_seed_find_is_stale", "_on_seed_find_error")
        f._on_seed_find_error("boom")
        self._assert_agrees(f)

    def test_the_find_found_nothing_path(self):
        f = self._parked()
        f._seed_find_token = 5
        f._seed_find_launch_token = 5
        f._seed_find_running = True
        f._sender = _TokenWorker(5)
        bind(f, "_seed_find_is_stale", "_on_seed_find_finished")
        f._on_seed_find_finished(_Report(markers=(), status="refused"))
        self._assert_agrees(f)

    def test_the_stop_waiting_path(self):
        f = self._parked()
        f._split_busy_kind = "find"
        f._seed_find_running = True
        bind(f, "_on_reshape_cancel")
        f._on_reshape_cancel()
        self._assert_agrees(f)

    def test_the_recipe_refusal_path(self):
        f = self._parked()
        f._reshape_recipe = lambda for_find=False: (
            None, "Pick the image channel the markers are in.", None)
        bind(f, "_find_reshape_markers")
        f._find_reshape_markers()
        self._assert_agrees(f)

    def test_a_successful_find_leaves_them_agreeing_too(self):
        """The ok path repopulates with real rows; a marker the user
        unticked by hand is preserved OFF, so it must STAY parked --
        agreement, not blanket un-parking."""
        f = self._parked(source=0)
        f._seed_find_token = 5
        f._seed_find_launch_token = 5
        f._seed_find_running = True
        f._sender = _TokenWorker(5)
        bind(f, "_seed_find_is_stale", "_on_seed_find_finished")
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        # 101 is not in the new rows at all, so nothing can un-park it
        # from the panel -- the sync has to.
        assert not f._parked_marker_ids()
        assert int((f._refine_layer.data == 101).sum()) > 0


class TestAMarkerKnobChangeRemovesTheMarkerLAYER:
    """The list emptied but `guide_markers` stayed drawn -- a picture of
    markers that no longer exist, beside a list that says there are none."""

    def test_the_orchestrator_removes_the_layer_and_the_ids(self):
        f = _HandlerFake()
        f._split_running = True
        bind(f, "_on_reshape_markers_invalidated",
             "_drop_materialised_markers",
             # U2b : the emptied list can no longer un-park
             # anything, so the handler puts every parked marker back.
             "_unpark_all_markers", "_parked_marker_ids",
             "_unpark_marker")
        f._remove_temp_layer = lambda key: f.removed_keys.append(key)
        f._on_reshape_markers_invalidated()
        assert f.removed_keys == ["markers"]

    def test_it_leaves_review_BEFORE_it_drops_the_markers(self):
        """MEASURED rather than read (audit L2).

        The snap-back writes `_refine_pre_preview` over the ROI, and
        that snapshot still holds the generation being invalidated -- so
        dropping the voxels first just paints them back, drawn, with no
        row and no bookkeeping entry.  The order is recorded as it
        actually happens, so a refactor that keeps both calls but swaps
        them still fails.
        """
        f = _HandlerFake()
        f._split_running = True
        order = []
        bind(f, "_on_reshape_markers_invalidated")
        f._remove_temp_layer = lambda key: order.append("remove:" + key)
        f._leave_reshape_review = lambda reason: order.append("leave")
        f._drop_materialised_markers = lambda **kw: order.append("drop")
        f._unpark_all_markers = lambda **kw: order.append("unpark")
        f._on_reshape_markers_invalidated()
        assert order.index("leave") < order.index("drop"), order
        assert order.index("drop") < order.index("unpark"), (
            "the drop pops its own stash entries, so a blanket un-park "
            "before it would resurrect the generation it just removed")

    def test_it_is_wired_into_the_callback_dict(self):
        import inspect

        src = inspect.getsource(_K()._open_advanced_features)
        assert '"on_markers_invalidated"' in src


class TestTheLabelsBuilderSaysWhyItGaveUp:
    def test_a_find_with_no_snapshot_writes_a_status_line(self):
        f = _HandlerFake()
        f._split_running = True
        f._split_roi = None
        bind(f, "_find_reshape_markers")
        f._reshape_recipe = lambda for_find=False: (object(), {}, None)
        f._reshape_labels_for_engine = lambda: None
        f._find_reshape_markers()
        texts = " ".join(str(c.get("status_text", "")) for c in f.ui_calls)
        assert texts.strip(), "it returned in silence"


class TestTeardownDropsEVERYTemporaryLayer:
    def test_the_registry_is_swept_on_teardown(self):
        import inspect

        src = inspect.getsource(_K()._teardown_reshape_for_switch)
        assert "_remove_all_temp_layers()" in src, (
            "a helper nothing calls is a helper that rots"
        )


# =========================================================================== #
# 22. Fix pass -- the launch recorder (UX-M7: runtime, not source-grep)
# =========================================================================== #
class _RecAdv(FakeAdv):
    """A FakeAdv that also answers the marker picker."""

    marker_ids = None

    def get_checked_marker_ids(self):
        return self.marker_ids

    def update_split_ui_state(self, **kw):
        self.ui_calls.append(kw)


class _Sig:
    def connect(self, *_a, **_k):
        pass

    def disconnect(self, *_a, **_k):
        pass


class _RecThread:
    started = _Sig()
    finished = _Sig()

    def start(self):
        pass

    def quit(self):
        pass

    def wait(self, _ms=0):
        return True


def _launch_recorder(**params):
    """A fake whose two launch paths record what reaches the worker.

    Both workers and QThread are replaced, so nothing runs -- the point is
    the ARGUMENTS, which is where a Find and a Preview can silently
    disagree.
    """
    import napari_epicycle.annotator.reshape_worker as RW
    import qtpy.QtCore as QtCore

    M = _M()
    vol = np.zeros((8, 8, 8), np.int32)
    vol[2:5, 2:5, 2:5] = 7
    roi = (slice(1, 6),) * 3
    base = dict(seeds_from=M.SEEDS_FROM_PAINTED, walls_from=M.WALLS_FROM_SHAPE,
                search_padding=15)
    base.update(params)
    f = FakeAnnotator(vol, roi, 7, adv=_RecAdv(base))
    f.viewer = type("V", (), {"layers": []})()
    f._refine_baseline = np.full((5, 5, 5), 7, np.int32)
    f._split_running = True
    f._split_seed_ids = [7, 101]

    main = FakeLayer(np.zeros((8, 8, 8), np.int32), "curation_3d_x")
    f._layer_by_name = lambda _layers, _name: main
    bind(f, "_reshape_labels_for_engine", "_find_reshape_markers",
         "_preview_reshape", "_reshape_recipe")

    # ONE array, produced once, handed to both.
    f.the_labels = f._reshape_labels_for_engine()
    f._reshape_labels_for_engine = lambda: f.the_labels

    spec = ReshapeSpec(target="label", seeds="paint", walls="distance")
    seeds = np.zeros((5, 5, 5), np.int32)
    seeds[0, 0, 0] = 101
    # Brick U1: a Find builds the recipe with `for_find=True` (the
    # source's own detector, no painted-addition promotion), a Preview
    # without it -- so the stub has to accept the kwarg.
    f._reshape_recipe = lambda for_find=False: (spec, {}, seeds)
    f._lock_subviewers_for_preview = lambda: None

    seen = []

    class _Rec:
        def __init__(self, labels, images, spec, **kw):
            kw.setdefault("keep_seed_ids", None)
            seen.append(dict(labels=labels, images=images, spec=spec, **kw))
            self.progress = _Sig()
            self.message = _Sig()
            self.finished = _Sig()
            self.error = _Sig()

        def moveToThread(self, _t):
            pass

        def run(self):
            pass

    RW.SeedFindWorker = _Rec
    RW.ReshapeWorker = _Rec
    QtCore.QThread = _RecThread
    return f, seen


@pytest.fixture(autouse=True)
def _restore_worker_classes():
    """Whatever `_launch_recorder` swapped out goes back."""
    import napari_epicycle.annotator.reshape_worker as RW
    import qtpy.QtCore as QtCore

    keep = (RW.SeedFindWorker, RW.ReshapeWorker, QtCore.QThread)
    yield
    RW.SeedFindWorker, RW.ReshapeWorker, QtCore.QThread = keep


class TestSearchPaddingIsASeam:
    """UX-M7: the knob read is its own method, so it is testable without
    driving the whole of Start."""

    def _fake(self, params=None, adv=True):
        f = FakeAnnotator(np.zeros((2, 2, 2), np.int32), (slice(0, 2),) * 3, 7,
                          adv=FakeAdv(params or {}))
        if not adv:
            f._adv = None
        bind(f, "_reshape_search_padding")
        return f

    def test_it_reads_the_knob(self):
        assert self._fake({"search_padding": 3})._reshape_search_padding() == 3

    def test_a_missing_key_is_todays_behaviour(self):
        f = self._fake({})
        f._adv._params.pop("search_padding", None)
        assert f._reshape_search_padding() == 15

    def test_no_panel_is_todays_behaviour(self):
        assert self._fake(adv=False)._reshape_search_padding() == 15

    def test_rubbish_is_todays_behaviour(self):
        assert self._fake({"search_padding": "x"})._reshape_search_padding() == 15

    def test_a_negative_value_is_clamped(self):
        assert self._fake({"search_padding": -4})._reshape_search_padding() == 0

    def test_start_uses_it_for_the_snapshot(self):
        import inspect

        src = inspect.getsource(_K()._start_split)
        assert "_reshape_search_padding()" in src
        assert "margin=_padding" in src


# =========================================================================== #
# 23. Fix pass (UX-M8) -- the finish handlers, driven for real
# =========================================================================== #
class TestTheFinishHandlersDriveThePanel:
    def _fake(self):
        f = _HandlerFake()
        f._adv = _RecAdv({})
        f._seed_find_token = 5
        f._seed_find_launch_token = 5
        f._seed_find_running = True
        f._split_label_id = 149
        bind(f, "_seed_find_is_stale", "_on_seed_find_finished",
             "_on_seed_find_error")
        f._signal_sender = lambda: _TokenWorker(5)
        f._remove_temp_layer = lambda key: f.removed_keys.append(key)
        f._seed_find_thread_cleanup = lambda: setattr(
            f, "cleanups", f.cleanups + 1)
        return f

    def _calls(self, f, key):
        return [c[key] for c in f._adv.ui_calls if key in c]

    def test_an_ok_report_fills_the_list_and_paints_the_markers(self):
        f = self._fake()
        f._on_seed_find_finished(_Report(markers=(1, 2, 3)))
        assert self._calls(f, "marker_ids") == [[1, 2, 3]]
        assert "materialised" in f.removed_keys, (
            "the markers have to land on the Refine layer -- that is the "
            "only layer the side views mirror"
        )

    def test_the_stats_carry_the_size_AND_the_place(self):
        """The detectors return single-voxel peaks, so the size alone is
        "1" for every marker; the position is what a user can act on."""
        f = self._fake()
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        stats = self._calls(f, "marker_stats")[0]
        assert set(stats) == {1, 2}
        assert stats[1]["voxels"] == 100
        assert stats[2]["centroid_zyx"] == (2.0, 4.0, 6.0)

    def test_the_busy_release_comes_AFTER_the_gate(self):
        f = self._fake()
        f._on_seed_find_finished(_Report(markers=(1,)))
        assert self._calls(f, "marker_busy") == [False]

    def test_a_refusal_empties_the_list_and_says_why(self):
        f = self._fake()
        f._on_seed_find_finished(
            _Report(status="refused", messages=["No markers found in this cell."]))
        assert self._calls(f, "marker_ids") == [[]]
        assert self._calls(f, "marker_note") == [
            "No markers found in this cell."]
        assert "markers" in f.removed_keys, "the drawn markers stayed"

    def test_a_refusal_leaves_Preview_alive(self):
        f = self._fake()
        f._on_seed_find_finished(_Report(status="refused"))
        assert any(c.get("preview_enabled") for c in f.ui_calls)

    def test_an_error_says_nothing_was_changed(self):
        f = self._fake()
        f._on_seed_find_error("boom")
        assert self._calls(f, "marker_ids") == [[]]
        note = self._calls(f, "marker_note")[0]
        assert "Nothing was changed" in note

    def test_a_cancelled_find_never_writes_the_list(self):
        f = self._fake()
        f._seed_find_running = False           # Stop already cleared it
        f._on_seed_find_finished(_Report(markers=(1, 2)))
        assert self._calls(f, "marker_ids") == []


# =========================================================================== #
# 24. Re-audit LOWs -- the panel is handed back on EVERY exit
# =========================================================================== #
class TestDiscardAlwaysHandsThePanelBack:
    """`_reject_split` has three exits.  Two of them -- the fragment-refine
    branch and the "volume not found" bail -- called `_reset_split_state()`
    but never `clear_split_guided_ui()`, so a Find in flight when Discard is
    pressed left the panel-local busy LATCHED: the Find button and the ten
    marker knobs greyed, the count line stuck on "Finding...", and nothing
    left on screen able to clear them."""

    def _fake(self, *, shared=None, fragment=False):
        vol = np.zeros((4, 4, 4), np.int32)
        f = FakeAnnotator(vol, (slice(0, 2),) * 3, 7, adv=_RecAdv({}))
        f.viewer = type("V", (), {"layers": []})()
        f.sync_manager = FakeSync({"curation_3d_x": shared})
        f._split_fragment_refine = fragment
        f._seed_find_running = True
        f._split_busy_kind = "find"
        f._smooth_running = False
        f._remove_prob_overlay = lambda: None
        f._reset_smooth_state = lambda: None
        f._set_selected_label_all = lambda _i: None
        f._on_isolation_changed = lambda _s: None
        f._clear_draw_snapshot = lambda: None
        f._split_isolate_was_on = False
        f._split_main_visible = True
        f.removed_keys = []
        bind(f, "_reject_split", "_fire_reshape_lock")
        # AFTER bind(): the shared binder installs the real registry helper.
        f._remove_temp_layer = lambda key: f.removed_keys.append(key)
        f._abandon_seed_find_worker = lambda: None
        f._abandon_reshape_worker = lambda: None
        return f

    def test_the_volume_not_found_bail_still_clears_the_panel(self):
        f = self._fake(shared=None)
        f._reject_split()
        assert f._adv.cleared == 1, (
            "Discard left the marker picker greyed with no way out"
        )

    def test_the_fragment_refine_branch_still_clears_the_panel(self):
        f = self._fake(shared=np.zeros((4, 4, 4), np.int32), fragment=True)
        f._reject_split()
        assert f._adv.cleared == 1

    def test_the_normal_path_still_clears_it_exactly_once(self):
        f = self._fake(shared=np.zeros((4, 4, 4), np.int32))
        f._reject_split()
        assert f._adv.cleared == 1, "the clear must not double-fire"

    def test_the_lock_still_comes_off_on_every_exit(self):
        for kwargs in ({"shared": None},
                       {"shared": np.zeros((4, 4, 4), np.int32),
                        "fragment": True},
                       {"shared": np.zeros((4, 4, 4), np.int32)}):
            f = self._fake(**kwargs)
            f._reshape_lock_fired = True        # as Start leaves it
            f._reject_split()
            assert f.lock_calls == [False], (
                "Step 3 stayed locked with the session already gone"
            )


class TestTheErrorHandlersHaveTheWasRunningLegToo:
    """The finish handlers drop a result that arrives after a Cancel; the
    error handlers did not, so a cancelled job's eventual crash still
    rewrote the panel the user had already moved on from."""

    def _find(self, running):
        f = _HandlerFake()
        f._adv = _RecAdv({})
        f._seed_find_token = 3
        f._seed_find_launch_token = 3
        f._seed_find_running = running
        bind(f, "_seed_find_is_stale", "_on_seed_find_error")
        f._signal_sender = lambda: _TokenWorker(3)
        f._seed_find_thread_cleanup = lambda: setattr(
            f, "cleanups", f.cleanups + 1)
        return f

    def _solve(self, running):
        f = _HandlerFake()
        f._adv = _RecAdv({})
        f._reshape_token = 3
        f._reshape_launch_token = 3
        f._reshape_running = running
        bind(f, "_reshape_is_stale", "_on_reshape_error")
        f._signal_sender = lambda: _TokenWorker(3)
        f._reshape_thread_cleanup = lambda: setattr(
            f, "cleanups", f.cleanups + 1)
        return f

    def test_a_find_error_after_a_cancel_says_nothing(self):
        f = self._find(running=False)
        f._on_seed_find_error("boom")
        assert f._adv.ui_calls == [], "a cancelled find still rewrote the panel"

    def test_a_live_find_error_does_speak(self):
        f = self._find(running=True)
        f._on_seed_find_error("boom")
        assert any("marker_note" in c for c in f._adv.ui_calls)

    def test_a_solve_error_after_a_cancel_says_nothing(self):
        f = self._solve(running=False)
        f._on_reshape_error("boom")
        assert f._adv.ui_calls == []

    def test_a_live_solve_error_does_speak(self):
        f = self._solve(running=True)
        f._on_reshape_error("boom")
        assert f.ui_calls, "a live failure must be visible"


# =========================================================================== #
# 25. The real flow: Start FIRST, choose the source SECOND
# =========================================================================== #
class TestGuidedIsDecidedLIVENotOnceAtStart:
    """Found by a real-button shots run, 2026-09-18.

    `_start_split` read `is_guided_split()` ONCE and latched
    `_reshape_active`.  But the flow a user actually takes is Start ->
    open Markers -> pick "Find them in an image channel" -> Find ->
    Preview: the source is chosen AFTER Start, so the flag was False and
    Preview ran the LEGACY painted-seed path, answering "Need >= 2 seed
    regions, found 0. Paint more seeds, then try again."

    The b2 driver missed it for a year of runs because it called
    `_preview_reshape()` directly instead of clicking the button.
    """

    def _started(self, **params):
        M = _M()
        vol = np.zeros((6, 6, 6), np.int32)
        vol[2:4, 2:4, 2:4] = 7
        refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
        base = dict(seeds_from=M.SEEDS_FROM_PAINTED,
                    walls_from=M.WALLS_FROM_SHAPE)
        base.update(params)
        f = FakeAnnotator(vol, (slice(0, 6),) * 3, 7, adv=FakeAdv(base),
                          refine=refine)
        f.viewer = type("V", (), {"layers": []})()
        f._refine_baseline = np.zeros((6, 6, 6), np.int32)
        f._split_running = True
        # ... and Start decided "legacy", because at that moment it was.
        f._reshape_active = False
        f.routed = []
        f._preview_reshape = lambda: f.routed.append("engine")
        f._refresh_reshape_inputs = lambda: f.routed.append("inputs")
        bind(f, "_preview_split", "_on_reshape_source_changed",
             "_fire_reshape_lock", "_arm_guided_if_needed",
             "_is_guided_now")
        return f, M

    def _choose(self, f, **params):
        """The user moves a picker after Start."""
        f._adv._params.update(params)
        f._on_reshape_source_changed(0)

    def test_an_image_marker_source_chosen_AFTER_start_still_routes(self):
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        f._preview_split()
        assert f.routed.count("engine") == 1, (
            "Preview ran the legacy painted-seed path and told the user to "
            "paint more seeds"
        )

    def test_a_membrane_wall_chosen_AFTER_start_still_routes(self):
        f, M = self._started()
        self._choose(f, walls_from=M.WALLS_FROM_MEMBRANE)
        f._preview_split()
        assert f.routed.count("engine") == 1

    def test_any_appended_wall_kind_arms_it_too(self):
        M = _M()
        for idx in range(1, len(M.WALL_KINDS)):
            f, _ = self._started()
            self._choose(f, walls_from=idx)
            assert f._reshape_active is True, M.WALL_KINDS[idx]

    def test_arming_late_puts_the_STEP3_LOCK_on(self):
        """The cell is punched out from Start; once the session becomes
        guided, Save and Commit Triage must stop being available."""
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        assert f._reshape_active is True
        assert f.lock_calls == [True]

    def test_arming_late_refreshes_the_inputs(self):
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        assert "inputs" in f.routed

    def test_arming_late_makes_PREVIEW_pressable(self):
        """Arming has to light Preview up, exactly as Start does.

        `_start_split`'s guided branch ends with
        `_update_split_ui(preview_enabled=True, ...)`, because a source
        that finds its own markers has nothing to paint and so never
        reaches the `>= 2 painted markers` rule that gates Preview on
        the legacy path.  Arming LATE did everything else -- the flag,
        the Step-3 lock, the inputs -- and not that, so in the order the
        user actually works (Start, THEN choose "Find them in an image
        channel") Preview stayed GREY.  Found by the live driver,
        check U1-01e, 2026-09-19: the button could not even be pressed.
        """
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        assert any(c.get("preview_enabled") is True for c in f.ui_calls), (
            "arming late left Preview disabled: {}".format(f.ui_calls))

    def test_arming_late_stops_telling_them_to_paint(self):
        """...and the line under it must stop asking for brush strokes.

        The status left over from Start reads "Painting marker 1 ...
        Paint inside the object in the 2D side views" -- an instruction
        for a source that does not take painted markers, sitting above a
        button the user cannot press.
        """
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        texts = [c["status_text"] for c in f.ui_calls if c.get("status_text")]
        assert texts, "arming late said nothing at all"
        assert "Press Preview" in texts[-1], texts[-1]

    def test_a_legacy_choice_after_start_does_not_touch_PREVIEW(self):
        """The mirror image: the legacy path's own gate is untouched.

        A source that IS painted still earns Preview by having two
        markers, so arming (which did not happen here) must not have
        pushed anything.
        """
        f, M = self._started()
        before = len(f.ui_calls)
        self._choose(f, seeds_from=M.SEEDS_FROM_DETECTED)
        assert f.ui_calls[before:] == [], f.ui_calls[before:]

    def test_it_arms_ONCE_however_many_pickers_move(self):
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        self._choose(f, walls_from=M.WALLS_FROM_MEMBRANE)
        assert f.lock_calls == [True]

    def test_nothing_is_armed_while_no_session_is_running(self):
        f, M = self._started()
        f._split_running = False
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        assert f._reshape_active is False
        assert f.lock_calls == []

    def test_a_legacy_choice_after_start_does_not_arm(self):
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_DETECTED)
        assert f._reshape_active is False
        assert f.lock_calls == []

    def test_preview_arms_it_itself_if_the_handler_never_ran(self):
        """Belt and braces: the picker's callback is wired through the
        panel, and a panel that was rebuilt mid-session would not have it."""
        f, M = self._started()
        f._adv._params["seeds_from"] = M.SEEDS_FROM_IMAGE   # no handler call
        assert f._reshape_active is False
        f._preview_split()
        assert f._reshape_active is True
        assert f.routed.count("engine") == 1
        assert f.lock_calls == [True]

    def test_switching_BACK_to_a_legacy_source_routes_legacy_again(self):
        """The lock stays armed (harmless -- the cell is still punched
        out), but Preview must follow the LIVE predicate."""
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        self._choose(f, seeds_from=M.SEEDS_FROM_PAINTED)
        f._preview_split()
        assert f.routed.count("engine") == 0, (
            "the engine ran for a hand-painted session"
        )

    def test_discard_after_a_late_arm_releases_the_lock(self):
        f, M = self._started()
        self._choose(f, seeds_from=M.SEEDS_FROM_IMAGE)
        f.sync_manager = FakeSync({"curation_3d_x": None})
        f._split_fragment_refine = False
        f._smooth_running = False
        f._clear_draw_snapshot = lambda: None
        f._split_isolate_was_on = False
        f._split_main_visible = True
        f.removed_keys = []
        bind(f, "_reject_split")
        f._remove_temp_layer = lambda key: f.removed_keys.append(key)
        f._abandon_seed_find_worker = lambda: None
        f._abandon_reshape_worker = lambda: None
        f._reject_split()
        assert f.lock_calls == [True, False]


# --------------------------------------------------------------------------- #
# Brick 2b -- the recipe carries every knob, and the quality gate is fed
# --------------------------------------------------------------------------- #
class TestTheRecipeCarriesEveryKnob:
    def test_every_new_knob_reaches_the_spec(self):
        fake, _M = _image_fake(
            seed_smooth=3.5, max_sigma_um=9.0, seed_thresh=0.35,
            seed_clip=0.05, seed_gamma=0.8, seed_bg=30, seed_denoise=4,
            smooth=2.5, refine_radius=9, refine_eps=0.05,
        )
        spec, _images, _seeds = fake._reshape_recipe()
        assert spec is not None
        p = spec.params
        assert p["seed_smooth"] == 3.5
        assert p["max_sigma_um"] == 9.0
        assert p["seed_thresh"] == 0.35
        assert p["seed_clip"] == 0.05
        assert p["seed_gamma"] == 0.8
        assert p["seed_bg"] == 30
        assert p["seed_denoise"] == 4
        assert p["smooth"] == 2.5
        assert p["refine_radius"] == 9
        assert p["refine_eps"] == 0.05

    def test_automatic_stays_None_rather_than_becoming_a_number(self):
        fake, _M = _image_fake(seed_thresh=None)
        spec, _i, _s = fake._reshape_recipe()
        assert spec.params["seed_thresh"] is None

    def test_the_wall_chain_runs_the_same_sizes_as_the_marker_chain(self):
        """One set of knobs: the wall chain either mirrors the
        marker chain outright or ticks the same steps, so two copies of
        lift / haze / despeckle could disagree with nothing on screen
        saying which won."""
        fake, _M = _image_fake(seed_gamma=0.8, seed_bg=30, seed_denoise=4)
        spec, _i, _s = fake._reshape_recipe()
        assert spec.params["wall_gamma"] == spec.params["seed_gamma"]
        assert spec.params["wall_bg"] == spec.params["seed_bg"]
        assert spec.params["wall_denoise"] == spec.params["seed_denoise"]

    def test_no_engine_param_is_left_at_a_default_the_panel_cannot_reach(self):
        """The asymmetry this brick closes: `DEFAULT_PARAMS` has 29 keys
        and the recipe used to name 13.  Everything except the ones the
        panel decides structurally is carried now."""
        from packages.epicycle_annotator.reshape.spec import DEFAULT_PARAMS

        fake, _M = _image_fake()
        spec, _i, _s = fake._reshape_recipe()
        # Set structurally by the recipe itself, or deliberately frozen.
        structural = {
            "seed_channel", "wall_channel", "seed_preproc", "wall_preproc",
            "size_budget", "arbiter_low_evidence_percentile",
            "arbiter_band_um",
        }
        unreached = sorted(
            k for k in DEFAULT_PARAMS
            if k not in structural and k not in spec.params
        )
        assert not unreached, unreached


class TestTheQualityGateIsFed:
    def _res(self, **kw):
        from packages.epicycle_annotator.reshape import ReshapeResult

        out = ReshapeResult()
        out.status = "ok"
        out.n_seeds = kw.pop("n_seeds", 2)
        out.n_pieces = kw.pop("n_pieces", 2)
        out.coverage = kw.pop("coverage", 1.0)
        sizes = kw.pop("sizes", [5000, 4800])
        out.piece_sizes = list(sizes)
        out.piece_stats = [
            {"id": 11 + i, "voxels": int(v), "centroid_zyx": (0, 0, 0),
             "bbox": (0, 0, 0, 1, 1, 1)}
            for i, v in enumerate(sizes)
        ]
        out.messages = list(kw.pop("messages", []))
        out.roi = (slice(0, 6),) * 3
        out.result = np.zeros((6, 6, 6), np.int32)
        for k, v in kw.items():
            setattr(out, k, v)
        return out

    def _fake(self, res):
        vol = np.zeros((6, 6, 6), np.int32)
        vol[2:4, 2:4, 2:4] = 149
        refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
        fake = FakeAnnotator(vol, res.roi, 149, adv=FakeAdv(), refine=refine)
        fake.next_id = 900
        bind(fake, "_apply_reshape_result",
         "_print_count_diag_result", "_render_reshape_preview",
             "_reshape_quality_notes", "_reshape_review_status")
        return fake

    def _notes_call(self, fake):
        for call in fake._adv.ui_calls:
            if "quality_notes" in call:
                return call["quality_notes"]
        raise AssertionError("the review never passed quality_notes")

    def test_a_clean_result_passes_an_empty_block(self):
        res = self._res()
        fake = self._fake(res)
        fake._apply_reshape_result(res)
        assert self._notes_call(fake) == []

    def test_a_low_coverage_result_reaches_the_panel(self):
        res = self._res(coverage=0.94)
        fake = self._fake(res)
        fake._apply_reshape_result(res)
        notes = self._notes_call(fake)
        assert [n.kind for n in notes] == ["coverage"]
        assert "object 149" in notes[0].text   # section 0a

    def test_the_sliver_note_names_the_FINAL_piece_ids(self):
        """The gate speaks in the engine's piece ids; the panel ticks the
        minted ones.  An untranslated id would untick nothing at all and
        read as "the rule does not work"."""
        res = self._res(n_seeds=3, n_pieces=3, sizes=[5000, 4800, 410])
        fake = self._fake(res)
        fake._apply_reshape_result(res)
        notes = [n for n in self._notes_call(fake) if n.kind == "sliver"]
        assert notes
        final = set(fake._reshape_final_ids.values())
        for note in notes:
            assert set(note.piece_ids) <= final, (note.piece_ids, final)

    def test_merge_to_count_suppresses_the_lost_marker_note(self):
        """The ENGINE's flag decides it, not the wording of its message
        (M2) -- so the fixture sets both, the way a real run does."""
        res = self._res(n_seeds=3, n_pieces=2, merged_to_count=True,
                        messages=["Merged 3 pieces down to the expected 2."])
        fake = self._fake(res)
        fake._apply_reshape_result(res)
        assert [n for n in self._notes_call(fake)
                if n.kind == "lost-marker"] == []

    def test_S1_and_the_block_agree_on_one_coverage_number(self):
        """"94 % assigned" and "6 % was not given" are one
        source formatted twice."""
        res = self._res(coverage=0.94)
        fake = self._fake(res)
        fake._apply_reshape_result(res)
        status = [str(c.get("status_text", "")) for c in fake.ui_calls
                  if c.get("status_text")]
        assert any("94 % assigned" in t for t in status), status
        note = self._notes_call(fake)[0]
        assert "6 %" in note.text

    def test_the_gate_never_takes_the_review_down_with_it(self):
        """A quality note is commentary.  If anything in the gate raises,
        the piece list still has to appear -- so the helper answers with
        an empty block rather than propagating."""
        res = self._res()
        fake = self._fake(res)

        class _Exploding:
            status = "ok"

            @property
            def coverage(self):
                raise RuntimeError("boom")

        assert fake._reshape_quality_notes(_Exploding()) == []
        fake._apply_reshape_result(res)
        assert self._notes_call(fake) == []


class TestSnapNeedsImageEvidenceToo:
    """M1: Snap is guarded exactly as the arbiter is.

    `refine_result_guided` falls back to the WALL elevation when there is
    no image one, so a stale tick with shape walls silently snapped every
    piece's edge onto the distance transform -- a real change to the
    committed geometry, with nothing on screen to explain it.  The widget
    unchecks the box when the walls stop being image-derived; this is the
    belt-and-braces behind it, the same one `arbiter` has had since R0.
    """

    def test_shape_walls_drop_the_refine_stage(self):
        M = _M()
        for kind in ("distance", "skeleton", "none"):
            fake, _ = _image_fake(walls_from=M.WALL_KINDS.index(kind),
                                  refine_guided=True)
            spec, _i, _s = fake._reshape_recipe()
            assert "guided" not in spec.refine, kind

    def test_every_image_wall_keeps_it(self):
        from packages.epicycle_annotator.reshape import IMAGE_DERIVED_WALLS

        M = _M()
        for idx, kind in enumerate(M.WALL_KINDS):
            if kind not in IMAGE_DERIVED_WALLS:
                continue
            fake, _ = _image_fake(walls_from=idx, refine_guided=True)
            spec, _i, _s = fake._reshape_recipe()
            assert "guided" in spec.refine, kind

    def test_it_is_the_same_rule_the_arbiter_uses(self):
        """One predicate, not two that can drift."""
        M = _M()
        for idx in range(len(M.WALL_KINDS)):
            fake, _ = _image_fake(walls_from=idx, refine_guided=True,
                                  arbiter=True)
            spec, _i, _s = fake._reshape_recipe()
            assert ("guided" in spec.refine) is bool(spec.arbiter), (
                M.WALL_KINDS[idx])

    def test_the_other_two_refine_stages_are_untouched(self):
        M = _M()
        fake, _ = _image_fake(walls_from=M.WALL_KINDS.index("distance"),
                              refine_guided=True, refine_merge_small=True,
                              min_size=20, target_count=3)
        spec, _i, _s = fake._reshape_recipe()
        assert "merge_small" in spec.refine
        assert "merge_to_count" in spec.refine


class TestTheGateReadsTheFlagNotThePose:
    """M2: rule (a)'s suppression used to be `"Merged " in messages`.

    Message text is user-facing prose this very fix pass is rewording, so
    the suppression was one wording change away from silently coming back
    on -- and it would have fired on any future message with that word in
    it.  The engine sets a flag now.
    """

    def _fake_with(self, **kw):
        from packages.epicycle_annotator.reshape import ReshapeResult

        res = ReshapeResult()
        res.status = "ok"
        res.n_seeds = 3
        res.n_pieces = 2
        res.coverage = 1.0
        res.piece_sizes = [5000, 4800]
        res.piece_stats = [{"id": 11, "voxels": 5000},
                           {"id": 12, "voxels": 4800}]
        res.roi = (slice(0, 6),) * 3
        res.result = np.zeros((6, 6, 6), np.int32)
        for k, v in kw.items():
            setattr(res, k, v)
        vol = np.zeros((6, 6, 6), np.int32)
        vol[2:4, 2:4, 2:4] = 149
        refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
        fake = FakeAnnotator(vol, res.roi, 149, adv=FakeAdv(), refine=refine)
        fake.next_id = 900
        bind(fake, "_reshape_quality_notes", "_reshape_review_status")
        return fake, res

    def test_the_flag_suppresses_rule_a(self):
        fake, res = self._fake_with(merged_to_count=True)
        fake._reshape_final_ids = {11: 901, 12: 902}
        assert [n for n in fake._reshape_quality_notes(res)
                if n.kind == "lost-marker"] == []

    def test_without_the_flag_the_rule_speaks(self):
        fake, res = self._fake_with(merged_to_count=False)
        fake._reshape_final_ids = {11: 901, 12: 902}
        assert [n for n in fake._reshape_quality_notes(res)
                if n.kind == "lost-marker"]

    def test_a_message_containing_the_word_no_longer_suppresses_anything(
            self):
        """The old rule's failure mode, pinned: prose must not decide."""
        fake, res = self._fake_with(
            merged_to_count=False,
            messages=["Merged regions were not touched by the arbiter."])
        fake._reshape_final_ids = {11: 901, 12: 902}
        assert [n for n in fake._reshape_quality_notes(res)
                if n.kind == "lost-marker"]

    def test_the_helper_no_longer_reads_the_messages_at_all(self):
        import inspect

        src = inspect.getsource(_K()._reshape_quality_notes)
        # CODE only: the comment recording the change quotes the old
        # substring, and a test that cannot tell a comment from a read
        # would fail on its own explanation.
        code = "".join(line for line in src.splitlines(True)
                       if not line.lstrip().startswith("#"))
        assert "Merged " not in code
        assert "merged_to_count" in code


class TestTheAmberBlockIsClearedWithTheResult(object):
    """M4: two paths cleared the pieces and left the warnings about them.

    Stop-then-Preview-again, and Refine Fragments, both push an empty
    piece list.  The quality block is ABOVE that list and described the
    pieces that are no longer there, so it kept explaining a result the
    user had already thrown away -- amber, and wrong.
    """

    def _fake(self):
        vol = np.zeros((6, 6, 6), np.int32)
        vol[2:4, 2:4, 2:4] = 149
        roi = (slice(0, 6),) * 3
        refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
        fake = FakeAnnotator(vol, roi, 149, adv=FakeAdv(), refine=refine)
        return fake

    def _cleared(self, fake):
        """Every call that pushed an empty piece list, and whether it
        cleared the block with it."""
        return [c for c in fake._adv.ui_calls
                if c.get("pieces") == []]

    def test_stop_clears_the_block_with_the_pieces(self):
        fake = self._fake()
        fake._reshape_running = True
        fake._reshape_worker = None
        fake._reshape_thread = None
        bind(fake, "_on_reshape_cancel", "_abandon_reshape_worker",
             "_reshape_thread_cleanup")
        fake._on_reshape_cancel()
        calls = self._cleared(fake)
        assert calls, "Stop pushed no empty piece list at all"
        assert all(c.get("quality_notes") == [] for c in calls), calls

    def test_the_source_of_both_clear_sites_passes_it(self):
        """`_enter_fragment_refine_mode` needs a live annotator to run,
        so its clear is read where it is written -- it is a literal
        argument list, which is exactly what source inspection is for."""
        import inspect

        src = inspect.getsource(_K()._enter_fragment_refine_mode)
        assert "quality_notes=[]" in src, (
            "Refine Fragments still leaves the amber block up over its "
            "fragment list"
        )


class TestEveryRefusalReachesTheRedLine(object):
    """M8: the five refusal doors all put their sentence in `engine_error`.

    They are ANSWERS (spec inv. 6), so each one has to be readable AS an
    answer -- not a grey line the user scrolls past looking for the
    result that never came.
    """

    def _preview_fake(self, **params):
        fake, _M = _image_fake(**params)
        fake._reshape_running = False
        fake._reshape_active = True
        fake._split_running = True
        bind(fake, "_preview_reshape")
        return fake

    def _errors(self, fake):
        return [c["engine_error"] for c in fake._adv.ui_calls
                if c.get("engine_error")]

    def test_a_blank_channel_is_red(self):
        fake = self._preview_fake(wall_channel="")
        fake._preview_reshape()
        errs = self._errors(fake)
        assert errs, fake._adv.ui_calls
        assert "walls" in errs[0].lower()

    def test_an_unknown_item_is_red(self):
        fake = self._preview_fake(seeds_from=99)
        fake._preview_reshape()
        assert self._errors(fake)

    def test_a_refused_RESULT_is_red_too(self):
        """The other door: the recipe was fine and the ENGINE refused
        (zero markers, over budget).  Same rule, same label."""
        from packages.epicycle_annotator.reshape import ReshapeResult

        res = ReshapeResult()
        res.status = "refused"
        res.refusal_reason = "no-seeds"
        res.messages = ["No markers found in this cell -- lower the "
                        "prominence."]
        vol = np.zeros((6, 6, 6), np.int32)
        vol[2:4, 2:4, 2:4] = 149
        roi = (slice(0, 6),) * 3
        refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
        fake = FakeAnnotator(vol, roi, 149, adv=FakeAdv(), refine=refine)
        fake._reshape_running = True
        bind(fake, "_on_reshape_finished", "_reshape_is_stale",
             "_reshape_thread_cleanup")
        fake._on_reshape_finished(res)
        errs = self._errors(fake)
        assert errs and "No markers" in errs[0], fake._adv.ui_calls

    def test_an_OK_result_clears_any_earlier_red(self):
        """A refusal left up over a successful preview is worse than one
        never shown."""
        import inspect

        src = inspect.getsource(_K()._apply_reshape_result)
        assert 'engine_error=""' in src


class TestAcceptCommitsWhatThePreviewProduced(object):
    """Accept must route on the last PREVIEW, not on a live flag.

    `_preview_split` routes on `_reshape_active AND _is_guided_now()`;
    `_accept_split` routed on `_reshape_active` alone.  The two disagree
    the moment the flag is armed after a legacy preview -- which
    `_arm_guided_if_needed` does on any picker move, and which the walls
    combo does to ITSELF when shape-thickness markers force it back to
    "shape only" (advanced_features_widget `_refresh_wall_item_availability`).

    Accept then called `_write_reshape_result`, which returns None
    because `_reshape_result` is None, and took the "nothing to apply"
    branch: the cell was RESTORED to its pre-Start shape and the user was
    told "There was nothing to apply".  A legitimate preview, accepted,
    silently reverted.
    """

    def _fake(self, *, reshape_active, reshape_result):
        vol = np.zeros((8, 8, 8), np.int32)
        vol[2:6, 2:6, 2:6] = 7
        roi = (slice(0, 8),) * 3
        refine = FakeLayer(np.zeros((8, 8, 8), np.int32), "Refine")
        fake = FakeAnnotator(vol, roi, 7, adv=FakeAdv(checked=[101, 102]),
                             refine=refine)
        fake._split_previewed = True
        fake._reshape_active = reshape_active
        fake._reshape_result = reshape_result
        fake._restored = []
        fake._restore_reshape_target = (
            lambda reason: fake._restored.append(reason))
        fake._locks = []
        fake._fire_reshape_lock = lambda on: fake._locks.append(bool(on))
        fake._wrote = []
        # the rest of the surface the LEGACY accept branch reads; the
        # reshape tests never reached it, so FakeAnnotator has no need
        # for these and they are set here rather than widened globally.
        fake._split_fragment_refine = False
        fake._split_isolate_was_on = False
        fake._split_seed_ids = [101, 102]
        fake._edt_overlay_layer = None
        fake._split_auto_smooth = False
        fake._teardown_reshape_for_switch = lambda *a, **k: None
        fake._split_main_visible = None
        fake._is_auto_advance_enabled = lambda: False
        fake._clear_draw_snapshot = lambda *a, **k: None
        fake._set_selected_label_all = lambda *a, **k: None
        fake.viewer = type("V", (), {"layers": []})()
        return fake, vol, refine

    def _legacy_preview_in_the_refine_layer(self, fake, refine, vol):
        """What a legacy preview leaves behind: the two pieces painted in
        the Refine layer and the target punched out of the volume."""
        refine.data[2:6, 2:4, 2:6] = 101
        refine.data[2:6, 4:6, 2:6] = 102
        vol[vol == 7] = 0

    def test_a_legacy_preview_is_NOT_thrown_away_when_the_flag_is_armed(
            self):
        """The bug, end to end: legacy preview, picker flipped, Accept."""
        fake, vol, refine = self._fake(
            reshape_active=True, reshape_result=None)
        self._legacy_preview_in_the_refine_layer(fake, refine, vol)
        bind(fake, "_accept_split", "_write_reshape_result")
        fake._accept_split()
        assert fake._restored == [], (
            "Accept restored the cell instead of committing the legacy "
            "preview: {}".format(fake._restored)
        )
        written = set(int(v) for v in np.unique(vol)) - {0}
        assert written == {101, 102}, (
            "the legacy pieces were not written: {}".format(sorted(written))
        )

    def test_and_the_lock_still_comes_off(self):
        fake, vol, refine = self._fake(
            reshape_active=True, reshape_result=None)
        self._legacy_preview_in_the_refine_layer(fake, refine, vol)
        bind(fake, "_accept_split", "_write_reshape_result")
        fake._accept_split()
        assert fake._locks and fake._locks[-1] is False, fake._locks

    def test_a_guided_preview_still_takes_the_engine_path(self, solved):
        """The other half: nothing changes when there IS a result."""
        labels, res = solved
        vol = labels.copy()
        vol[vol == 7] = 0
        refine = FakeLayer(np.zeros_like(labels), "Refine")
        piece_ids = [int(r["id"]) for r in res.piece_stats]
        finals = {p: 100 + i for i, p in enumerate(piece_ids)}
        fake = FakeAnnotator(
            vol, res.roi, 7,
            adv=FakeAdv(checked=list(finals.values())), refine=refine)
        fake._reshape_result = res
        fake._reshape_final_ids = finals
        fake._split_previewed = True
        fake._reshape_active = True
        fake._split_fragment_refine = False
        bind(fake, "_write_reshape_result")
        out = fake._write_reshape_result(vol, 7)
        assert out is not None
        assert out["new_ids"], out
        assert set(out["new_ids"]) <= set(np.unique(vol))

    def test_the_two_routers_agree_on_the_same_question(self):
        """Stated once, so the pair cannot drift again: whatever decides
        the PREVIEW's path, the ACCEPT's path is decided by what that
        preview actually produced."""
        import inspect

        accept = inspect.getsource(_K()._accept_split)
        code = "".join(line for line in accept.splitlines(True)
                       if not line.lstrip().startswith("#"))
        assert "_reshape_result" in code, (
            "_accept_split still routes on a flag rather than on the "
            "result the last preview left"
        )

    def test_an_armed_flag_with_no_preview_at_all_is_still_refused(self):
        """Not previewed -> Accept does nothing, as before."""
        fake, vol, refine = self._fake(
            reshape_active=True, reshape_result=None)
        fake._split_previewed = False
        before = vol.copy()
        bind(fake, "_accept_split", "_write_reshape_result")
        fake._accept_split()
        assert np.array_equal(vol, before)


# =========================================================================== #
# 27. Fix pass M4 -- "object", not "cell", in what the Reshape flow says
# =========================================================================== #
class TestTheReshapeFlowSaysObject:
    """UX round : the tab is used on nuclei as well as cells, so the
    thing being reshaped is an OBJECT everywhere the user reads it.

    S1 and S2 were converted; the five status lines the WORKER handlers
    push were not, so every cancel, every error and every empty Find
    answered in the old vocabulary -- and those are exactly the moments
    a user stops to read the line.
    """

    HANDLERS = (
        "_on_seed_find_finished",
        "_on_reshape_cancel",
        "_on_reshape_error",
        "_on_reshape_finished",
        "_enter_fragment_refine_mode",
    )

    @pytest.mark.parametrize("name", HANDLERS)
    def test_no_handler_string_says_cell(self, name):
        import inspect
        import re

        fn = getattr(_K(), name, None)
        if fn is None:
            pytest.skip("{} does not exist".format(name))
        src = inspect.getsource(fn)
        # Strings only: a COMMENT may still say cell, and several do.
        offenders = [
            lit for lit in re.findall(r'"([^"\n]*)"', src)
            if re.search(r"\bcells?\b", lit)
        ]
        assert not offenders, (
            "{} still speaks the old vocabulary: {}".format(name, offenders))

    def test_the_engine_refusal_says_object_too(self):
        """The line `_on_seed_find_finished` falls back to, and the one
        `reshape_region` refuses a seedless cut with, are read side by
        side with the panel's own -- one saying "cell" and the other
        "object" for the same thing is worse than either."""
        from packages.epicycle_annotator.reshape import engine

        import inspect

        src = inspect.getsource(engine.reshape_region)
        assert "No markers found in this object" in src
        assert "lie outside this object" in src

        # The SAME two sentences live a second time in `find_seeds`, and
        # `_on_seed_find_finished` surfaces `messages[-1]` verbatim -- so a
        # Find that comes back empty is read on the same panel line.
        find_src = inspect.getsource(engine.find_seeds)
        assert "No markers found in this object" in find_src
        assert "lie outside this object" in find_src
        assert "this cell" not in find_src

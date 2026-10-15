"""Chaining ACROSS methods (decided 2026-09-23).

B1 disabled Preview whenever the method combo moved away from the method
that staged the result (the old state 4m).  The owner dropped that rule:
chaining is the feature, so a method switch keeps Preview live.  B1c
(decided 2026-09-23) went further: ANY method chains onto any other,
in any order, with any settings -- the Settle crossing is gone, and a later
step that needs more room GROWS the region (and its snapshot, exactly)
instead of being refused.  What is left to refuse: a worker in flight, a
synchronous step inside its pump, a different object, a missing channel.

Allowing the cross-method chain opened a hazard this file pins: a refusal
that fires AFTER the chain count is bumped but before the solve (Guided
Grow with no probability map, a missing channel, a failed wall picture)
used to call `_reset_smooth_state`, which does NOT restore the snapshot --
so the first step's voxels stayed in the volume with Accept and Discard
gone and the lock released.  These tests DRIVE the real `_preview_smooth`
on a bare fake; they never read source.
"""

from __future__ import annotations

import numpy as np
import pytest


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


_BOUND = (
    "_preview_smooth", "_finish_refine_preview", "_reset_smooth_state",
    "_fire_refine_lock", "_refine_breadcrumb_text", "_refine_spacing",
    "_refine_spacing_known", "_abandon_refine_worker",
    "_refine_thread_cleanup", "_drop_refine_walls",
    "_drop_refine_walls_layer", "_accept_smooth", "_reject_smooth",
    "_refuse_refine_step", "_on_refine_method_combo_changed",
    "_refresh_refine_preview_gate", "_launch_refine_method",
    "_launch_refine_settle", "_push_still_staged",
    "_refine_still_staged_text", "_settle_roi_margin",
    "_contour_roi_margin", "_launch_refine_contour",
    "_refine_room", "_refine_step_margin", "_grow_refine_region",
    "_finish_settle_result",
    "_refine_staged_status", "_refine_s1_text",
)
_CLASS_TABLES = ("_REFINE_NOTHING_MOVED", "_REFINE_STEP_WORD",
                 "_REFINE_METHOD_WORD")

#: combo index per engine key -- REFINE_METHOD_KEYS order.
_IDX = {"sdf": 0, "contour": 1, "dilate_erode": 2, "guided_grow": 3,
        "fill_holes": 4, "flood_fill": 5, "settle": 6}


class _Val:
    def __init__(self, v):
        self._v = v

    def value(self):
        return self._v

    def currentText(self):
        return self._v

    def currentIndex(self):
        return self._v


class _Adv:
    """The panel, reduced to what `_preview_smooth` reads."""

    def __init__(self):
        self.method = "sdf"
        self.channel = ""
        self.wall_kind = "membrane"
        self.band_um = 2.0
        self.refine_de_radius_sb = _Val(1)
        self.refine_de_axis_combo = _Val("3D")
        self.refine_gg_max_radius_sb = _Val(5)
        self.refine_gg_prob_threshold_sb = _Val(0.5)
        self.refine_gg_axis_combo = _Val("3D")
        self.refine_fh_axis_combo = _Val("3D")

    @property
    def refine_method_combo(self):
        return _Val(_IDX[self.method])

    def refine_method_key(self):
        return self.method

    def get_refine_params(self):
        needs = self.method in ("contour", "settle")
        return {"needs_channel": needs, "channel": self.channel,
                "wall_kind": self.wall_kind, "band_um": self.band_um}


class _Sync:
    def __init__(self, vol):
        self.shared_labels = {"labels": vol}


class _Layer:
    selected_label = 7


class _Viewer:
    layers = ()


class Fake:
    def __init__(self, vol):
        self._active_labels_name = "labels"
        self.sync_manager = _Sync(vol)
        self.viewer = _Viewer()
        self.adv = _Adv()
        self.ui_calls = []
        self.lock_calls = []

        self._split_running = False
        self._split_fragment_refine = False
        self._batch_staged = False
        self._batch_running = False
        self._smooth_running = False
        self._smooth_snapshot = None
        self._smooth_roi = None
        self._smooth_label_id = None
        self._smooth_chain_count = 0
        self._refine_steps = []
        self._refine_staged_method = None
        self._refine_running = False
        self._refine_lock_fired = False
        self._refine_stopping = False
        self._refine_worker = None
        self._refine_thread = None
        self._refine_elevation = None
        self._refine_elevation_roi = None
        self._refine_pending_elevation = None
        self._refine_walls_wanted = False
        self.draw_calls = []             # "lock" / "restore", in order
        self._temp_guide_layers = {}
        self._interp_refine_mode = False
        self._load_locked = False

        K = _K()
        for name in _BOUND:
            setattr(self, name, getattr(K, name).__get__(self, Fake))
        for name in _CLASS_TABLES:
            setattr(self, name, getattr(K, name))
        self._snapshot_label_roi = K._snapshot_label_roi
        self._label_bbox_roi = K._label_bbox_roi
        self._refine_lock_cb = self.lock_calls.append

    # --- seams ---------------------------------------------------------
    def _update_smooth_ui(self, **kw):
        self.ui_calls.append(kw)

    def _get_advanced_features_widget(self):
        return self.adv

    def _ensure_shared_data_aligned(self, name):
        return self.sync_manager.shared_labels.get(name)

    def _layer_by_name(self, layers, name):
        return _Layer()

    def _get_prob_roi(self, roi):
        return None                      # this patch has no probability map

    def _get_refine_channel_roi(self, roi):
        return None

    def _refresh_refine_object_row(self):
        pass

    def _sync_all_views_after_operation(self, roi=None, label_id=None):
        pass

    def _refresh_refine_walls_toggle(self):
        pass

    def _lock_subviewers_for_preview(self):
        self.draw_calls.append("lock")

    def _restore_ready_to_draw_state(self):
        self.draw_calls.append("restore")

    def _refresh_merge_button_state(self):
        pass

    def _refresh_label_action_buttons_state(self):
        pass

    def _remove_temp_layer(self, key):
        self._temp_guide_layers.pop(key, None)

    def _reshape_context(self):
        return {"patch_name": "p", "run_epoch": 0, "spacing": (3.0, 1.0, 1.0)}

    def _remove_prob_overlay(self):
        pass

    def _clear_draw_snapshot(self):
        pass

    def _is_auto_advance_enabled(self):
        return False

    def last(self, key):
        for kw in reversed(self.ui_calls):
            if key in kw:
                return kw[key]
        return None


@pytest.fixture(autouse=True)
def _qt(monkeypatch):
    from qtpy.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    import napari_epicycle.annotator.curation_3d_widget as mod

    monkeypatch.setattr(mod, "Labels", _Layer)


def _vol():
    v = np.zeros((28, 32, 32), np.int32)
    v[6:22, 8:24, 8:24] = 7
    v[6:22, 8:11, 8:11] = 0           # a notch, so Smooth has work to do
    return v


def _staged_smooth():
    vol = _vol()
    before = vol.copy()
    f = Fake(vol)
    f.adv.method = "sdf"
    f._preview_smooth()
    assert f._smooth_running and f._refine_staged_method == "sdf"
    assert not np.array_equal(vol, before), "Smooth wrote nothing"
    return f, vol, before


class TestAMethodSwitchKeepsPreviewLive:
    def test_switching_to_dilate_erode_leaves_preview_enabled(self):
        f, _vol_, _b = _staged_smooth()
        f.adv.method = "dilate_erode"
        f._on_refine_method_combo_changed(2)
        assert f.last("preview_enabled") is True

    def test_and_the_chain_runs(self):
        f, vol, before = _staged_smooth()
        f.adv.method = "dilate_erode"
        f._on_refine_method_combo_changed(2)
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert f._refine_steps == ["Smooth", "Dilate/Erode"]
        assert f.last("accept_text") == "Accept (2)"
        f._reject_smooth()
        assert np.array_equal(vol, before)

    def test_settle_at_the_default_band_is_offered(self):
        """B1c: Settle fits inside the 15 voxels Smooth reserved."""
        f, _v, _b = _staged_smooth()
        f.adv.method = "settle"
        f.adv.channel = "membrane"
        f._on_refine_method_combo_changed(6)
        assert f.last("preview_enabled") is True

    def test_a_settle_with_a_wide_band_is_offered_too(self):
        """B1c: no refusal before the click -- the region grows at it."""
        f, _v, _b = _staged_smooth()
        f.adv.method = "settle"
        f.adv.channel = "membrane"
        f.adv.band_um = 20.0              # 20 voxels at 1 um xy > 15
        f._on_refine_method_combo_changed(6)
        assert f.last("preview_enabled") is True

    def test_a_channel_method_with_no_channel_disables_preview(self):
        f, _v, _b = _staged_smooth()
        f.adv.method = "contour"
        f._on_refine_method_combo_changed(1)
        assert f.last("preview_enabled") is False
        assert f.last("preview_reason") == "Pick a channel first."

    def test_switching_back_restores_the_staged_status(self):
        """The status the Preview WROTE comes back, not a generic "a
        result is staged" (B1b audit UX-L2)."""
        f, _v, _b = _staged_smooth()
        written = f.last("status_text")
        assert "voxels changed" in written
        f.adv.method = "contour"
        f._on_refine_method_combo_changed(1)
        assert f.last("status_text") == "Pick a channel first."
        f.adv.method = "sdf"
        f._on_refine_method_combo_changed(0)
        assert f.last("preview_enabled") is True
        assert f.last("status_text") == written


class TestARefusalMidChainKeepsTheStagedResult:
    """The hazard: Guided Grow with no probability map, chained onto a
    staged Smooth."""

    def test_the_smooth_result_is_still_staged(self):
        f, vol, before = _staged_smooth()
        staged_vol = vol.copy()
        f.adv.method = "guided_grow"
        f._preview_smooth()
        assert f._smooth_running is True
        assert f._smooth_snapshot is not None
        assert f._smooth_chain_count == 1
        assert f._refine_staged_method == "sdf"
        assert f._refine_lock_fired is True
        assert np.array_equal(vol, staged_vol)
        assert f.last("accept_enabled") is True
        assert f.last("reject_enabled") is True
        assert "probability map" in f.last("status_text")

    def test_discard_still_restores_the_snapshot(self):
        f, vol, before = _staged_smooth()
        f.adv.method = "guided_grow"
        f._preview_smooth()
        f._reject_smooth()
        assert np.array_equal(vol, before)
        assert f._refine_lock_fired is False

    def test_a_missing_channel_at_the_click_keeps_it_too(self):
        f, vol, before = _staged_smooth()
        f.adv.method = "contour"            # needs a channel, none chosen
        f._preview_smooth()
        assert f._smooth_running and f._smooth_snapshot is not None
        assert f._smooth_chain_count == 1
        f._reject_smooth()
        assert np.array_equal(vol, before)

    def test_a_failed_wall_picture_mid_chain_keeps_it_too(self):
        f, vol, before = _staged_smooth()
        staged_vol = vol.copy()
        f.adv.method = "contour"
        f.adv.channel = "membrane"          # chosen, but gone from the viewer
        f._preview_smooth()
        assert f._smooth_running and f._smooth_snapshot is not None
        assert f._smooth_chain_count == 1
        assert np.array_equal(vol, staged_vol)
        f._reject_smooth()
        assert np.array_equal(vol, before)

    def test_on_a_first_preview_the_session_still_resets(self):
        vol = _vol()
        f = Fake(vol)
        f.adv.method = "guided_grow"
        f._preview_smooth()
        assert f._smooth_running is False
        assert f._smooth_snapshot is None
        assert f._smooth_chain_count == 0


# --------------------------------------------------------------------------- #
# B1c (decided 2026-09-23): the chaining rule is ROOM-based.  Settle
# moves voxels only between background and the object, and its region is at
# least the shared 15, so it chains like any other method.
# --------------------------------------------------------------------------- #
def _elev(shape):
    e = np.zeros(shape, np.float32)
    e[:, 5, 5:27] = 1.0
    e[:, 26, 5:27] = 1.0
    e[:, 5:27, 5] = 1.0
    return e


def _sync_settle(f):
    """Run Settle synchronously: the worker is the only thing replaced."""
    from packages.epicycle_annotator.reshape.settle import settle_label

    K = _K()
    f._finish_settle_result = K._finish_settle_result.__get__(f, Fake)
    f._refine_band_um = lambda: float(f.adv.band_um)
    full = _elev(f.sync_manager.shared_labels["labels"].shape)

    def _launch(method, params, roi, roi_data, label_id):
        assert method == "settle"
        # as `_on_refine_finished` does with the worker's picture (B3): it
        # WAITS as the pending picture until the step writes
        f._refine_pending_elevation = (full[roi], tuple(roi))
        # as `_on_refine_finished` does before the result handler
        f._restore_ready_to_draw_state()
        res = settle_label(np.array(roi_data), full[roi], label_id,
                           band_um=float(params.get("band_um", 2.0)),
                           spacing=f._refine_spacing())
        f._finish_settle_result(res, label_id, roi)

    f._launch_refine_method = _launch
    return f


def _staged_settle(band_um=2.0):
    vol = _vol()
    before = vol.copy()
    f = _sync_settle(Fake(vol))
    f.adv.method = "settle"
    f.adv.channel = "membrane"
    f.adv.band_um = band_um
    f._preview_smooth()
    assert f._smooth_running and f._refine_staged_method == "settle"
    assert not np.array_equal(vol, before), "Settle wrote nothing"
    return f, vol, before


class TestAnyMethodChainsOntoAnyOther:
    def test_every_first_step_records_its_margin(self):
        f, _v, _b = _staged_smooth()
        assert f._smooth_roi_margin == 15
        g, _v2, _b2 = _staged_settle(2.0)
        assert g._smooth_roi_margin == g._settle_roi_margin({"band_um": 2.0})

    def test_settle_then_smooth_chains(self):
        f, vol, before = _staged_settle()
        f.adv.method = "sdf"
        f._on_refine_method_combo_changed(0)
        assert f.last("preview_enabled") is True
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert f._refine_steps == ["Settle", "Smooth"]
        assert f.last("accept_text") == "Accept (2)"
        f._reject_smooth()
        assert np.array_equal(vol, before), "Discard is not byte-identical"

    def test_smooth_then_settle_at_the_default_band_chains(self):
        f, vol, before = _staged_smooth()
        _sync_settle(f)
        f.adv.method = "settle"
        f.adv.channel = "membrane"
        f.adv.band_um = 2.0
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert f._refine_steps == ["Smooth", "Settle"]
        f._reject_smooth()
        assert np.array_equal(vol, before)


# --------------------------------------------------------------------------- #
# B1c: a later step that needs more room GROWS the region, exactly.  The
# volume is big enough that the growth is not clipped away by its walls.
# --------------------------------------------------------------------------- #
def _big_vol():
    v = np.zeros((40, 120, 120), np.int32)
    v[10:30, 35:85, 35:85] = 7
    v[10:30, 35:40, 35:40] = 0        # a notch
    return v


def _big_staged_settle(band_um=2.0):
    vol = _big_vol()
    before = vol.copy()
    f = _sync_settle(Fake(vol))
    f.adv.method = "settle"
    f.adv.channel = "membrane"
    f.adv.band_um = band_um
    f._preview_smooth()
    assert f._smooth_running and f._refine_staged_method == "settle"
    return f, vol, before


def _inside(inner, outer):
    return all(o.start <= i.start and i.stop <= o.stop
               for i, o in zip(inner, outer))


class TestTheRegionGrows:
    def test_c_l4_settle_small_then_settle_big_chains(self):
        """Owner C-L4, kept as a test of the GROWTH path: the chained big
        Settle equals a fresh big Settle on the accepted small result."""
        f, vol, before = _big_staged_settle(2.0)
        roi1 = f._smooth_roi
        after_small = vol.copy()
        f.adv.band_um = 20.0
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert f._refine_steps == ["Settle", "Settle"]
        assert f.last("accept_text") == "Accept (2)"
        assert f._smooth_roi_margin == 20
        assert _inside(roi1, f._smooth_roi) and f._smooth_roi != roi1
        chained = vol.copy()

        fresh_vol = after_small.copy()
        g = _sync_settle(Fake(fresh_vol))
        g.adv.method, g.adv.channel, g.adv.band_um = "settle", "membrane", 20.0
        g._preview_smooth()
        assert np.array_equal(fresh_vol, chained)

        f._reject_smooth()
        assert np.array_equal(vol, before), "Discard is not byte-identical"

    def test_smooth_then_a_wide_settle_grows_and_chains(self):
        vol = _big_vol()
        before = vol.copy()
        f = _sync_settle(Fake(vol))
        f.adv.method = "sdf"
        f._preview_smooth()
        roi1 = f._smooth_roi
        f.adv.method, f.adv.channel, f.adv.band_um = "settle", "membrane", 20.0
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert f._refine_steps == ["Smooth", "Settle"]
        assert _inside(roi1, f._smooth_roi) and f._smooth_roi != roi1
        f._reject_smooth()
        assert np.array_equal(vol, before)

    def test_the_grown_snapshot_is_exact(self):
        f, vol, before = _big_staged_settle(2.0)
        old_roi, old_snap = f._smooth_roi, f._smooth_snapshot.copy()
        f._grow_refine_region(vol, 7, 20)
        new_roi = f._smooth_roi
        inner = tuple(slice(o.start - n.start, o.stop - n.start)
                      for o, n in zip(old_roi, new_roi))
        assert np.array_equal(f._smooth_snapshot[inner], old_snap)
        assert np.array_equal(f._smooth_snapshot, before[new_roi])

    def test_discard_restores_the_ring_only_the_grown_region_covers(self):
        f, vol, before = _big_staged_settle(2.0)
        old_roi = f._smooth_roi
        f._grow_refine_region(vol, 7, 20)
        new_roi = f._smooth_roi
        ring = np.zeros(vol.shape, bool)
        ring[new_roi] = True
        ring[old_roi] = False
        assert ring.any()
        vol[ring] = 7                    # a later step wrote into the ring
        f._reject_smooth()
        assert np.array_equal(vol, before)

    def test_accept_hands_the_controller_the_grown_undo_region(self):
        f, vol, before = _big_staged_settle(2.0)
        got = {}
        f._on_label_smoothed_cb = (
            lambda lid, roi_slices=None, old_roi_data=None, **_kw:
            got.update(roi=roi_slices, snap=old_roi_data))
        f.adv.band_um = 20.0
        f._preview_smooth()
        grown = f._smooth_roi
        f._accept_smooth()
        assert got["roi"] == grown
        assert np.array_equal(got["snap"], before[grown])


class _WallsAdv(_Adv):
    def __init__(self):
        super().__init__()
        self.walls = []
        self.checked = []

    def set_refine_walls_toggle_enabled(self, enabled, reason=""):
        self.walls.append((bool(enabled), reason))

    def set_refine_walls_toggle_checked(self, checked):
        self.checked.append(bool(checked))

    def clear_refine_walls_toggle(self):
        self.walls.append((False, ""))


class TestAChainedNonWallStepDropsTheWallPicture:
    def test_settle_then_smooth(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            WALLS_TOGGLE_LAST_STEP,
        )

        f, _vol_, _b = _staged_settle()
        adv = _WallsAdv()
        adv.method, adv.channel = "settle", "membrane"
        f.adv = adv
        f._refresh_refine_walls_toggle = (
            _K()._refresh_refine_walls_toggle.__get__(f, Fake))
        f._show_walls_picture = lambda: True
        f._refine_walls_wanted = True
        f._temp_guide_layers["walls"] = object()
        assert f._refine_elevation is not None
        adv.method = "sdf"
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert f._refine_elevation is None
        assert "walls" not in f._temp_guide_layers
        assert adv.walls[-1] == (False, WALLS_TOGGLE_LAST_STEP)
        assert adv.checked and adv.checked[-1] is False


# --------------------------------------------------------------------------- #
# B1c audit fixes
# --------------------------------------------------------------------------- #
class TestAChainedNoOpRelocksTheSideViews:
    """MED-1: a worker's finish handler unlocks the side views; a chained
    step that then writes nothing must lock them again, or a stroke lands
    on the staged result outside what Discard and the growth account for."""

    def _chain_bumped(self):
        f, vol, before = _staged_smooth()
        f._smooth_chain_count = 2
        f.draw_calls.clear()
        f._restore_ready_to_draw_state()     # `_on_refine_finished`
        return f, vol, before

    def test_settle_refused_mid_chain(self):
        from types import SimpleNamespace

        f, _v, _b = self._chain_bumped()
        f._finish_settle_result(
            SimpleNamespace(status="refused", messages=["too big"]),
            7, f._smooth_roi)
        assert f._smooth_chain_count == 1
        assert f.draw_calls[-1] == "lock"

    def test_nothing_moved_mid_chain(self):
        f, vol, _b = self._chain_bumped()
        roi = f._smooth_roi
        f._finish_refine_preview("settle", np.array(vol[roi]), 0, "m", 7, roi)
        assert f._smooth_chain_count == 1
        assert f.draw_calls[-1] == "lock"

    def test_settle_empty_write_refused_mid_chain(self):
        f, vol, _b = self._chain_bumped()
        roi = f._smooth_roi
        f._finish_refine_preview(
            "settle", np.zeros_like(vol[roi]), 5, "m", 7, roi)
        assert f._smooth_chain_count == 1
        assert f.draw_calls[-1] == "lock"

    def test_driven_a_refused_wide_settle_after_smooth(self):
        f, _v, _b = _staged_smooth()
        _sync_settle(f)
        f.adv.method, f.adv.channel, f.adv.band_um = "settle", "membrane", 20.0
        f.draw_calls.clear()
        f._preview_smooth()                  # the object is too thin: refused
        assert f._smooth_chain_count == 1
        assert "restore" in f.draw_calls
        assert f.draw_calls[-1] == "lock"


class _Btn:
    def setEnabled(self, _v):
        pass

    def setText(self, _v):
        pass


class TestInterpDiscardRestoresTheGrownRegion:
    """MED-2: in interp-refine mode Discard put back only `_interp_roi`;
    a grown chain's writes beyond it survived."""

    def test_writes_beyond_the_interp_roi_are_undone(self):
        vol = _big_vol()
        original = vol.copy()
        f = _sync_settle(Fake(vol))
        # the interpolation: it wrote inside its own ROI
        iroi = (slice(10, 30), slice(35, 60), slice(35, 60))
        f._interp_roi = iroi
        f._interp_snapshot = vol[iroi].copy()
        vol[iroi][vol[iroi] == 0] = 7
        f._interp_label_id = 7
        f._interp_refine_mode = True
        for name in ("_interpolate_btn", "_accept_interp_btn",
                     "_reject_interp_btn", "_interp_refine_btn",
                     "_interp_status_label"):
            setattr(f, name, _Btn())
        f.adv.method = "sdf"
        f._preview_smooth()
        f._grow_refine_region(vol, 7, 25)
        ring = np.zeros(vol.shape, bool)
        ring[f._smooth_roi] = True
        ring[iroi] = False
        vol[ring] = 9                        # a grown step wrote out here
        f._reject_smooth()
        assert np.array_equal(vol, original)


class TestGrowthOnAnEmptyObject:
    """LOW-1: no growth, and no whole-volume copy."""

    def test_deleted_object_is_not_grown(self):
        f, vol, _b = _big_staged_settle(2.0)
        roi, snap = f._smooth_roi, f._smooth_snapshot
        vol[vol == 7] = 0

        def _boom(*_a, **_k):
            raise AssertionError("the volume was copied")

        f._snapshot_label_roi = _boom
        f._grow_refine_region(vol, 7, 25)
        assert f._smooth_roi == roi
        assert f._smooth_snapshot is snap


class TestTheStagedPictureSurvivesANoOpStep:
    """LOW-2: a worker step's picture is kept aside until it WRITES."""

    def test_a_refused_settle_keeps_the_staged_settles_picture(self):
        f, _v, _b = _big_staged_settle(2.0)
        elev, eroi = f._refine_elevation, f._refine_elevation_roi
        assert elev is not None
        f.adv.band_um = 20.0
        vol = f.sync_manager.shared_labels["labels"]
        # thin the object so the wide band is refused on Settle's own rule
        vol[vol == 7] = 0
        vol[18:21, 55:58, 55:58] = 7
        f._preview_smooth()
        assert f._refine_elevation is elev
        assert f._refine_elevation_roi == eroi

    def test_a_contour_that_moved_nothing_keeps_it(self):
        f, vol, _b = _big_staged_settle(2.0)
        elev = f._refine_elevation
        f._smooth_chain_count = 2
        f._refine_pending_elevation = (None, None)      # an ACWE launch
        roi = f._smooth_roi
        f._finish_refine_preview("contour", np.array(vol[roi]), 0, "m", 7,
                                 roi)
        assert f._refine_elevation is elev

    def test_a_contour_that_wrote_without_a_picture_clears_it(self):
        f, vol, _b = _big_staged_settle(2.0)
        f._smooth_chain_count = 2
        f._refine_pending_elevation = (None, None)
        roi = f._smooth_roi
        res = np.array(vol[roi])
        res[res == 7] = 0
        res[10:14, 20:30, 20:30] = 7
        f._finish_refine_preview("contour", res, int(np.count_nonzero(
            res != vol[roi])), "m", 7, roi)
        assert f._refine_elevation is None

    def test_a_settle_that_wrote_takes_its_own_picture(self):
        f, _v, _b = _big_staged_settle(2.0)
        assert f._refine_elevation is not None
        assert f._refine_pending_elevation is None


# --------------------------------------------------------------------------- #
# The auditor's probe cases: growth at the volume's edges and on an object
# that shrank or vanished mid-chain.
# --------------------------------------------------------------------------- #
def _corner():
    v = np.zeros((30, 40, 40), np.int32)
    v[0:8, 0:10, 0:10] = 7
    v[0:8, 0:3, 0:3] = 0
    return v


def _far_corner():
    v = np.zeros((30, 40, 40), np.int32)
    v[22:30, 30:40, 30:40] = 7
    v[22:30, 30:33, 30:33] = 0
    return v


def _slab():
    v = np.zeros((40, 80, 80), np.int32)
    v[10:30, 20:60, 20:60] = 7
    v[10:30, 20:25, 20:25] = 0
    return v


def _no_change(vol, roi):
    pass


def _shrink(vol, roi):
    m = np.zeros(vol.shape, bool)
    m[roi] = True
    vol[m & (vol == 7)] = 0
    vol[18:20, 38:40, 38:40] = 7


def _delete(vol, roi):
    m = np.zeros(vol.shape, bool)
    m[roi] = True
    vol[m & (vol == 7)] = 0


@pytest.mark.parametrize("make, mutate", [
    (_corner, _no_change), (_far_corner, _no_change),
    (_slab, _shrink), (_slab, _delete),
], ids=["origin-corner", "far-corner", "shrunk", "deleted"])
def test_growth_is_exact(make, mutate):
    vol = make()
    before = vol.copy()
    f = Fake(vol)
    f.adv.method = "sdf"
    f._preview_smooth()
    old = f._smooth_roi
    mutate(vol, old)                     # every write inside the old ROI
    f._grow_refine_region(vol, 7, 25)
    new = f._smooth_roi
    assert _inside(old, new)
    assert np.array_equal(f._smooth_snapshot, before[new])
    f._reject_smooth()
    assert np.array_equal(vol, before)

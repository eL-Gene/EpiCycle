"""The Refine lock and the three lifecycle holes the B1 audit found.

Everything here DRIVES the real methods on a bare fake rather than reading
their source, because all three failures are about ORDER and about which
branch ran -- exactly what `inspect.getsource` cannot see:

* **C-H1** a staged result from one of the five SYNCHRONOUS methods never
  raised the controller lock, because `_fire_refine_lock(True)` lived only
  in `_launch_refine_worker`.  Save Curation, Commit Triage and a run switch
  were all live over an unaccepted shape that is already in ``shared_data``
  -- and a run switch runs `_guard_unsaved_changes()` first, so answering
  "Save" would have PERSISTED it;
* **C-H2** a worker error on a FIRST (unchained) preview locked Step 3 for
  the rest of the session: `_smooth_chain_count` is 1, so `> 1` did not
  decrement it and `<= 0` never reset the session -- Accept and Discard both
  dead, `_refine_running` true for ever;
* **C-H3** `_accept_smooth` abandoned the worker, dropped the wall picture,
  cleared the staged method and RELEASED THE LOCK before its two early
  returns -- so an Accept refused for being load-locked threw the session's
  scaffolding away and left the shape staged with no lock on it.
"""

from __future__ import annotations

import numpy as np
import pytest


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


#: Everything the three paths under test actually touch.  Bound from the
#: REAL class, so a rename or a reordered line fails here.
_BOUND = (
    "_finish_refine_preview", "_reset_smooth_state", "_fire_refine_lock",
    "_refine_breadcrumb_text", "_refine_spacing", "_refine_spacing_known",
    "_abandon_refine_worker", "_refine_thread_cleanup", "_drop_refine_walls",
    "_drop_refine_walls_layer", "_on_refine_error", "_on_refine_stopped",
    "_accept_smooth", "_refine_is_stale", "_result_token", "_signal_sender",
    "_refresh_refine_preview_gate", "_push_still_staged",
    "_refine_still_staged_text",
)

#: The class-level sentence tables the methods read off ``self``.
_CLASS_TABLES = ("_REFINE_NOTHING_MOVED", "_REFINE_STEP_WORD",
                 "_REFINE_METHOD_WORD")


class _Sync:
    def __init__(self, vol):
        self.shared_labels = {"labels": vol}


class Fake:
    """A bare annotator: no Qt, no napari, no viewer."""

    def __init__(self, vol, **kw):
        self._active_labels_name = "labels"
        self.sync_manager = _Sync(vol)
        self.ui_calls = []
        self.lock_calls = []
        self.synced = []
        self.locked_subviewers = 0
        self.restored = 0

        self._smooth_running = True
        self._smooth_snapshot = vol.copy()
        self._smooth_roi = (slice(None), slice(None), slice(None))
        self._smooth_label_id = 7
        self._smooth_chain_count = 1
        self._refine_steps = []
        self._refine_staged_method = None
        self._refine_running = False
        self._refine_lock_fired = False
        self._refine_stopping = False
        self._refine_busy_kind = None
        self._refine_worker = None
        self._refine_thread = None
        self._refine_elevation = None
        self._refine_elevation_roi = None
        self._refine_walls_wanted = False
        self._temp_guide_layers = {}
        self._interp_refine_mode = False
        self._load_locked = False
        self._refine_token = 1
        self._refine_launch_labels_name = "labels"
        self._refine_launch_patch = "p"
        self._refine_launch_run_epoch = 0
        self.__dict__.update(kw)

        K = _K()
        for name in _BOUND:
            setattr(self, name, getattr(K, name).__get__(self, Fake))
        for name in _CLASS_TABLES:
            setattr(self, name, getattr(K, name))
        # the lock callback the controller would install
        self._refine_lock_cb = self.lock_calls.append

    # --- the seams the methods under test call -----------------------
    def _update_smooth_ui(self, **kw):
        self.ui_calls.append(kw)

    def _sync_all_views_after_operation(self, roi=None, label_id=None):
        self.synced.append((roi, label_id))

    def _refresh_refine_walls_toggle(self):
        pass

    def _lock_subviewers_for_preview(self):
        self.locked_subviewers += 1

    def _restore_ready_to_draw_state(self):
        self.restored += 1

    def _refresh_merge_button_state(self):
        pass

    def _refresh_label_action_buttons_state(self):
        pass

    def _remove_temp_layer(self, key):
        self._temp_guide_layers.pop(key, None)

    def _get_advanced_features_widget(self):
        return None

    def _reshape_context(self):
        return {"patch_name": "p", "run_epoch": 0, "spacing": (3.0, 1.0, 1.0)}

    def sender(self):
        return None

    def _remove_prob_overlay(self):
        pass

    def _clear_draw_snapshot(self):
        pass

    def _is_auto_advance_enabled(self):
        return False

    def last_ui(self, key):
        for kw in reversed(self.ui_calls):
            if key in kw:
                return kw[key]
        return None


def _vol():
    v = np.zeros((6, 12, 12), np.int32)
    v[1:5, 3:9, 3:9] = 7
    return v


def _grown(vol, label_id=7):
    """A legal one-voxel growth of the object, as a result ROI."""
    out = np.zeros_like(vol)
    out[1:5, 2:10, 3:9] = label_id
    return out


# --------------------------------------------------------------------------- #
# C-H1
# --------------------------------------------------------------------------- #
class TestASyncStepRaisesTheControllerLock:
    """The staged window is the same window whichever method opened it: the
    shape is in `shared_data` and has not been accepted."""

    def test_a_staged_sync_step_raises_it(self):
        vol = _vol()
        f = Fake(vol)
        f._finish_refine_preview("sdf", _grown(vol), 48, "48 voxels changed.",
                                 7, f._smooth_roi)
        assert f.lock_calls == [True], f.lock_calls
        # `_refine_lock_fired` is the ANNOTATOR's record of the lock it
        # asked for.  Its own `_refine_running` means "a worker is in
        # flight" and is correctly False here -- the CONTROLLER's flag of
        # the same name is the one `lock_calls` drives.
        assert f._refine_lock_fired is True

    @pytest.mark.parametrize("method", ["sdf", "dilate_erode", "guided_grow",
                                        "fill_holes", "flood_fill"])
    def test_every_synchronous_method_raises_it(self, method):
        vol = _vol()
        f = Fake(vol)
        f._finish_refine_preview(method, _grown(vol), 48, "changed.", 7,
                                 f._smooth_roi)
        assert f.lock_calls == [True]

    def test_it_is_raised_AFTER_the_write_not_before(self):
        """A lock taken before a write that then refuses would leave the
        surface locked over nothing."""
        vol = _vol()
        f = Fake(vol)
        f._finish_refine_preview("sdf", _grown(vol), 48, "changed.", 7,
                                 f._smooth_roi)
        assert f.synced, "the write did not happen"
        assert f.lock_calls == [True]

    def test_a_step_that_changed_nothing_does_NOT_raise_it(self):
        vol = _vol()
        f = Fake(vol)
        f._finish_refine_preview("fill_holes", _grown(vol), 0, "no holes.", 7,
                                 f._smooth_roi)
        assert f.lock_calls == [False] or f.lock_calls == []
        assert f._refine_lock_fired is False

    def test_accept_releases_it(self):
        vol = _vol()
        f = Fake(vol)
        f._finish_refine_preview("sdf", _grown(vol), 48, "changed.", 7,
                                 f._smooth_roi)
        f._on_label_smoothed_cb = None
        f._accept_smooth()
        assert f.lock_calls[-1] is False
        assert f._refine_lock_fired is False

    def test_it_is_idempotent_across_a_chain(self):
        vol = _vol()
        f = Fake(vol)
        f._finish_refine_preview("sdf", _grown(vol), 48, "a", 7, f._smooth_roi)
        f._smooth_chain_count = 2
        f._finish_refine_preview("dilate_erode", _grown(vol), 12, "b", 7,
                                 f._smooth_roi)
        assert f.lock_calls == [True], "the second step re-fired the lock"


# --------------------------------------------------------------------------- #
# C-H2
# --------------------------------------------------------------------------- #
class TestAnErrorOnTheFirstPreviewDoesNotStrandStepThree:
    def test_the_session_is_reset(self):
        vol = _vol()
        f = Fake(vol, _refine_running=True, _refine_lock_fired=True,
                 _refine_staged_method=None)
        f._smooth_chain_count = 1
        f._on_refine_error("boom")
        assert f._smooth_running is False
        assert f._smooth_snapshot is None
        assert f._smooth_roi is None
        assert f._smooth_chain_count == 0

    def test_the_controller_lock_comes_off(self):
        vol = _vol()
        f = Fake(vol, _refine_running=True, _refine_lock_fired=True)
        f._smooth_chain_count = 1
        f._on_refine_error("boom")
        assert f.lock_calls[-1] is False
        assert f._refine_lock_fired is False
        assert f._refine_running is False

    def test_accept_and_discard_are_not_left_dead(self):
        vol = _vol()
        f = Fake(vol, _refine_lock_fired=True)
        f._smooth_chain_count = 1
        f._on_refine_error("boom")
        assert f.last_ui("preview_enabled") is True

    def test_drawing_is_unlocked(self):
        vol = _vol()
        f = Fake(vol, _refine_lock_fired=True)
        f._smooth_chain_count = 1
        f._on_refine_error("boom")
        assert f.restored >= 1

    def test_a_CHAINED_error_keeps_the_earlier_steps(self):
        vol = _vol()
        f = Fake(vol, _refine_lock_fired=True)
        f._smooth_chain_count = 3
        f._on_refine_error("boom")
        assert f._smooth_running is True
        assert f._smooth_chain_count == 2
        assert f.lock_calls == [], "a chained error must not drop the lock"

    def test_the_stale_branch_still_joins_the_thread(self):
        """L4: `_on_refine_finished`'s stale branch calls
        `_refine_thread_cleanup`; this one did not, so a superseded run
        left its QThread in the slot."""
        vol = _vol()
        f = Fake(vol)
        f._refine_launch_patch = "somewhere else"     # -> stale

        class _T:
            quit_calls = 0

            def quit(self):
                _T.quit_calls += 1

            def wait(self, _ms):
                return True

        f._refine_thread = _T()
        f._on_refine_error("boom")
        assert f._refine_thread is None
        assert _T.quit_calls == 1


# --------------------------------------------------------------------------- #
# C-H3
# --------------------------------------------------------------------------- #
class TestAcceptDoesNotTearDownBeforeItsEarlyReturns:
    def test_a_load_locked_accept_leaves_the_session_intact(self):
        vol = _vol()
        f = Fake(vol, _load_locked=True)
        f._finish_refine_preview("sdf", _grown(vol), 48, "changed.", 7,
                                 f._smooth_roi)
        f.lock_calls.clear()
        f._accept_smooth()
        assert f._smooth_running is True, "the session was torn down"
        assert f._refine_staged_method == "sdf"
        assert f._refine_steps == ["Smooth"]
        assert f.lock_calls == [], "the lock was released over a staged shape"
        assert f._refine_lock_fired is True

    def test_an_accept_with_nothing_staged_changes_nothing(self):
        vol = _vol()
        f = Fake(vol, _smooth_running=False)
        f._accept_smooth()
        assert f.lock_calls == []

    def test_a_real_accept_still_tears_down(self):
        vol = _vol()
        f = Fake(vol)
        f._finish_refine_preview("sdf", _grown(vol), 48, "changed.", 7,
                                 f._smooth_roi)
        f._on_label_smoothed_cb = None
        f._accept_smooth()
        assert f._smooth_running is False
        assert f._refine_staged_method is None
        assert f._refine_steps == []
        assert f.lock_calls[-1] is False


# --------------------------------------------------------------------------- #
# C-M1, after Brick B2 -- the contour's limit is APPLIED, by its own knob
# --------------------------------------------------------------------------- #
class TestTheContourIsLimitedByItsOwnKnob:
    """B1 applied an INTERIM limit in the shared tail -- 8 voxels of the
    finest axis, the retired `Dist clamp`'s default -- because its contour
    page had no micron knob.  B2 puts "Edge may move" on the page and the
    two-sided limit inside `refine.contour_refine`, so the tail stops
    clipping and the number the user set is the number that binds."""

    def _big(self):
        v = np.zeros((10, 44, 44), np.int32)
        v[2:8, 7:37, 7:37] = 7
        return v

    def test_the_interim_voxel_constant_is_gone(self):
        import napari_epicycle.annotator.curation_3d_widget as M

        assert not hasattr(M, "REFINE_INTERIM_LIMIT_VOXELS")

    def test_the_call_carries_the_page_limit_and_the_patch_spacing(self):
        from packages.epicycle_annotator.refine import contour_with_walls

        vol = self._big()
        f = Fake(vol)
        f._refine_contour_call = _K()._refine_contour_call.__get__(f, Fake)
        # a flat probability map: no channel, a picture with no wall in it
        # (so only the push moves the outline), built by the worker-side
        # wrapper (B2 audit C-M3)
        fn, kw = f._refine_contour_call(
            {"contour_mode": 0, "contour_limit_um": 4.5,
             "wall_kind": "probability",
             "contour_balloon": 1.0, "contour_iterations": 30},
            vol.copy(), 7, None, np.zeros(vol.shape, np.float32))
        assert fn is contour_with_walls
        assert kw["limit_um"] == 4.5
        assert kw["spacing"] == (3.0, 1.0, 1.0)
        res = fn(**kw)
        from scipy.ndimage import distance_transform_edt

        d_out = distance_transform_edt(vol != 7, sampling=(3.0, 1.0, 1.0))
        grew = (res[0] == 7) & (vol != 7)
        assert grew.any()
        assert float(d_out[grew].max()) <= 4.5 + 1e-6
        assert res.details["hit_limit"] is True

    def test_the_tail_does_not_clip_a_second_time(self):
        """Only the ROI and the fence stand between a result and the
        volume now: the tail writes what the method returned."""
        vol = self._big()
        f = Fake(vol)
        grown = np.where(self._big() == 7, 7, 0)
        grown[1:9, 5:39, 5:39] = 7            # 2 voxels out everywhere
        f._finish_refine_preview("contour", grown, 999, "changed.", 7,
                                 f._smooth_roi)
        assert int((vol == 7).sum()) == int((grown == 7).sum())

    def test_hitting_the_limit_is_said_out_loud(self):
        vol = self._big()
        f = Fake(vol)
        f._finish_refine_preview(
            "contour", np.where(self._big() == 7, 7, 0), 5, "changed.", 7,
            f._smooth_roi,
            extra_warning="The outline reached the 3.0 um limit on one side.")
        warn = f.last_ui("result_warning") or ""
        assert "limit" in warn.lower(), warn

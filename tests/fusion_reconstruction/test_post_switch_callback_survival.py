"""After a patch switch, an Accept still reaches the controller.

The controller half of the 2026-09-19 regression (the widget half is
`tests/annotator/test_curation_callbacks_partial_set.py`).
`_post_switch_annotator_sync` re-wires the annotator with three Reshape
kwargs; `set_curation_callbacks` used to null everything it was not
handed.  The user-visible cost was not a crash -- it was a Smooth or a
Batch Refine that LOOKED accepted, changed the volume in memory, and was
thrown away by the next patch load because no dirty flag was ever set.

So the assertion here is the whole chain, not the wiring alone: switch,
accept, and then ask whether the controller knows there is something to
save.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


def _C():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    return FusionReconstructionController


# --------------------------------------------------------------------------- #
# the two fakes: an annotator that only holds callbacks, and a controller
# that only records what its handlers were told
# --------------------------------------------------------------------------- #
class _FakeAnnotator:
    """Everything `_post_switch_annotator_sync` touches on the widget."""

    def __init__(self):
        cls = _K()
        # the real method under test
        self.set_curation_callbacks = (
            cls.set_curation_callbacks.__get__(self, _FakeAnnotator))
        self._fire_batch_refine_lock = (
            cls._fire_batch_refine_lock.__get__(self, _FakeAnnotator))
        for attr in (
            "_on_interpolation_accepted_cb", "_on_label_split_cb",
            "_on_label_smoothed_cb", "_on_recalculate_cb",
            "_on_new_label_painted_cb", "_on_batch_refined_cb",
            "_batch_refine_lock_cb", "_add_mode_lock_cb",
            "_on_add_committed_cb", "_on_add_undo_session_cb",
            "_save_batch_recipe_cb", "_get_saved_batch_recipes_cb",
            "_reshape_lock_cb", "_on_reshape_committed_cb",
            "_reshape_context_cb",
            "_save_refine_recipe_cb", "_get_saved_refine_recipes_cb",
            "_delete_refine_recipe_cb",
        ):
            setattr(self, attr, None)
        self._active_labels_name = "curation_3d_x"
        self._active_image_name = "img"
        self.prob_path = None
        # the rest of the surface `_post_switch_annotator_sync` touches
        self._batch_refine_dialog = None
        self._suppress_layer_removal_cleanup = True
        self.layer_manager = type("_LM", (), {"_suppress_events": True})()
        self.focused = None
        self.deferred = []

    def set_focused_refinement(self, ids, centroids):
        self.focused = (list(ids), centroids)

    def _schedule_deferred_auto_load(self, ms):
        self.deferred.append(ms)

    # the rest of what the sync helper calls
    def set_prob_volume_path(self, p):
        self.prob_path = p

    def set_triage_callbacks(self, **kw):
        pass

    def set_runtime_callbacks(self, **kw):
        pass

    def refresh_triage_options(self, *a, **k):
        pass


class _FakeController:
    """Records which handler was called, and whether the edit was noticed."""

    def __init__(self, widget):
        self._widget = widget
        self._curation_dirty = False
        self._committed = []
        self._batch_locks = []

    # -- the handlers the launcher wires --------------------------------
    def _on_label_smoothed(self, *a, **k):
        self._commit_curation_action("smooth")

    def _on_label_split(self, *a, **k):
        self._commit_curation_action("split")

    def _on_interpolation_accepted(self, *a, **k):
        self._commit_curation_action("interpolate")

    def _on_batch_refined(self, *a, **k):
        self._commit_curation_action("batch_refine")

    def _on_recalculate(self, *a, **k):
        self._commit_curation_action("recalculate")

    def _on_new_label_painted(self, *a, **k):
        self._commit_curation_action("new_label")

    def _on_batch_refine_lock(self, running):
        self._batch_locks.append(bool(running))

    def _commit_curation_action(self, kind):
        self._committed.append(kind)
        self._curation_dirty = True

    # -- what `_post_switch_annotator_sync` reaches for ------------------
    def _on_annotator_refine_lock(self, *a, **k):
        # Brick B1: the post-switch re-wire passes FOUR
        # callbacks now, and a lock that is wired at launch but not after a
        # patch switch is a lock that silently stops working on the second
        # patch.
        pass

    def _on_annotator_reshape_lock(self, *a, **k):
        pass

    def _on_annotator_reshape_committed(self, *a, **k):
        pass

    def _on_annotator_reshape_context(self, *a, **k):
        return {}

    def _get_active_prob_path(self):
        return None

    def _get_curation_annotator_widget(self):
        return self._widget

    def _collect_flagged_labels(self):
        return [], {}

    def _on_annotator_mark_tier(self, *a, **k):
        pass

    def _on_annotator_label_deleted(self, *a, **k):
        pass

    def _on_annotator_label_merged(self, *a, **k):
        pass

    def _add_label_id_floor(self):
        # The post-switch re-wire passes the id floor
        return 0


#: The launcher's full wiring, by keyword.
def _wire_everything(widget, ctrl):
    widget.set_curation_callbacks(
        on_interpolation_accepted=ctrl._on_interpolation_accepted,
        on_label_split=ctrl._on_label_split,
        on_label_smoothed=ctrl._on_label_smoothed,
        on_recalculate=ctrl._on_recalculate,
        on_new_label_painted=ctrl._on_new_label_painted,
        on_batch_refined=ctrl._on_batch_refined,
        on_batch_refine_lock=ctrl._on_batch_refine_lock,
        on_reshape_lock=ctrl._on_annotator_reshape_lock,
        on_refine_lock=ctrl._on_annotator_refine_lock,
        on_reshape_committed=ctrl._on_annotator_reshape_committed,
        on_reshape_context=ctrl._on_annotator_reshape_context,
    )


def _switch(widget, ctrl):
    """Exactly what the patch switch does to the callbacks."""
    widget.set_curation_callbacks(
        on_reshape_lock=ctrl._on_annotator_reshape_lock,
        on_refine_lock=ctrl._on_annotator_refine_lock,
        on_reshape_committed=ctrl._on_annotator_reshape_committed,
        on_reshape_context=ctrl._on_annotator_reshape_context,
    )


# --------------------------------------------------------------------------- #
class TestAnAcceptStillReachesTheControllerAfterASwitch:
    def _wired(self):
        w = _FakeAnnotator()
        c = _FakeController(w)
        _wire_everything(w, c)
        return w, c

    def test_smooth_accept_marks_the_workspace_dirty_after_a_switch(self):
        """The headline case: Refine > SDF Smooth > Accept.  Before the
        fix the callback was None, so the volume changed and nothing
        else did -- and the next patch load discarded it."""
        w, c = self._wired()
        _switch(w, c)
        assert w._on_label_smoothed_cb is not None, (
            "the patch switch unwired Smooth"
        )
        w._on_label_smoothed_cb(412)
        assert c._committed == ["smooth"]
        assert c._curation_dirty is True

    def test_batch_refine_accept_too(self):
        w, c = self._wired()
        _switch(w, c)
        assert w._on_batch_refined_cb is not None
        w._on_batch_refined_cb([1, 2], "recipe")
        assert c._curation_dirty is True

    @pytest.mark.parametrize("attr,kind", [
        ("_on_interpolation_accepted_cb", "interpolate"),
        ("_on_label_split_cb", "split"),
        ("_on_label_smoothed_cb", "smooth"),
        ("_on_recalculate_cb", "recalculate"),
        ("_on_new_label_painted_cb", "new_label"),
        ("_on_batch_refined_cb", "batch_refine"),
    ])
    def test_every_one_of_the_six_commit_paths_survives(self, attr, kind):
        w, c = self._wired()
        _switch(w, c)
        cb = getattr(w, attr)
        assert cb is not None, "{} was nulled by the switch".format(attr)
        cb(1)
        assert c._committed == [kind]
        assert c._curation_dirty is True

    def test_the_batch_refine_LOCK_still_reaches_the_controller(self):
        """Not a commit path, but the same wiping: a lock that no longer
        fires leaves Step 3 unlocked during a batch refine."""
        w, c = self._wired()
        _switch(w, c)
        w._fire_batch_refine_lock(True)
        w._fire_batch_refine_lock(False)
        assert c._batch_locks == [True, False]

    def test_two_switches_in_a_row_are_still_fine(self):
        w, c = self._wired()
        _switch(w, c)
        _switch(w, c)
        assert w._on_label_smoothed_cb is not None
        w._on_label_smoothed_cb(1)
        assert c._curation_dirty is True

    def test_the_switch_still_wires_the_three_it_DOES_pass(self):
        """The guard must not turn the partial call into a no-op."""
        w = _FakeAnnotator()
        c = _FakeController(w)
        _switch(w, c)
        # `==` not `is`: attribute access on a bound method builds a new
        # wrapper every time, and what we are asking is "same function,
        # same controller", which is exactly what `==` compares.
        assert w._reshape_lock_cb == c._on_annotator_reshape_lock
        assert w._on_reshape_committed_cb == (
            c._on_annotator_reshape_committed)
        assert w._reshape_context_cb == c._on_annotator_reshape_context


class TestTheRealSyncHelperDoesTheSameThing:
    """The fakes above model the call; this runs the production method."""

    def test_post_switch_annotator_sync_leaves_the_commit_paths_wired(self):
        w = _FakeAnnotator()
        c = _FakeController(w)
        _wire_everything(w, c)

        helper = _C()._post_switch_annotator_sync.__get__(c, _FakeController)
        helper(w, "stack_patch_1")

        # it really ran: the flags it resets are reset, and it scheduled
        # the deferred auto-load it always schedules
        assert w._suppress_layer_removal_cleanup is False
        assert w.layer_manager._suppress_events is False
        assert w.deferred == [50]

        assert w._on_label_smoothed_cb is not None, (
            "_post_switch_annotator_sync unwired Smooth"
        )
        w._on_label_smoothed_cb(1)
        assert c._curation_dirty is True

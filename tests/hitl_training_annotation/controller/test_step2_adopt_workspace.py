"""Step 2 `adopt_workspace` — the shared-workspace hand-over.

Adopt runs the QUIET half of Browse — put the path in the box, clear the
previous context's scalars, load the state, refresh the three lists and the
run picker — and **none** of the Browse-only half: no timer flush on a virgin
step, no migration modals, no first-run modal, no saved-patch auto-load, and
no publish back up to the shell.

The call-order recorder below is the point of the file: it proves the
partition in both directions, so a later edit that quietly moves an item
across the line fails here rather than on the dev PC.

Qt-free: the controller is built with `__new__` and stub collaborators, the
same style as the rest of `tests/hitl_training_annotation/controller/`.
"""
from __future__ import annotations

import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (  # noqa: E501
    TrainingAnnotationController,
)

#: Everything Browse does that adopt must NOT do (table, right column).
BROWSE_ONLY = (
    "_flush_pending_state_timers",
    "_close_step2_annotator_dock_safely",
    "_clear_patch_viewer_layers_safely",
    "_maybe_prompt_step2_migration",
    "_maybe_prompt_split_to_flat_migration",
    "_auto_reload_saved_patch",
    "_maybe_prompt_first_run",
    "_publish_workspace_to_shell",
)

#: Everything the quiet half does, which BOTH paths must perform.
QUIET = (
    "_load_internal_state",
    "_refresh_round_status",
    "_refresh_sample_queue",
    "_refresh_patch_list_badges",
    "_refresh_run_picker",
)


class _Line:
    def __init__(self, text=""):
        self._text = text
        self.sets = []

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value
        self.sets.append(value)


class _Bar:
    def __init__(self):
        self.ranges = []

    def setRange(self, lo, hi):
        self.ranges.append((lo, hi))

    def setValue(self, v):
        pass


class _UI:
    def __init__(self):
        self.set_workdir_path = _Line()
        self.loaded_patch_label = _Line("stale patch")
        self.patch_busy_bar = _Bar()


class _SegManager:
    def __init__(self):
        self.segmenter = object()


def _controller(calls, working_dir=""):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = working_dir
    ctrl.active_run_id = "stale_run"
    ctrl.loaded_stack_path = "C:/old/patches"
    ctrl.loaded_patch_name = "old_patch"
    ctrl.sample_records = {"a": 1}
    ctrl.training_data_by_round = {"r": 1}
    ctrl.losses_by_round = {"r": [1]}
    ctrl.model_checkpoint_paths = {"r": "x"}
    ctrl.round_backend_info = {"r": {}}
    ctrl.segmentation_manager = _SegManager()
    ctrl._suppress_publish = False
    ctrl._workflow_shell = None
    ctrl._busy_depth = 0
    ctrl._state_save_timer = object()   # a pending timer exists, so the flush is reachable
    ctrl.logs = []
    ctrl._append_ui_log = lambda msg: ctrl.logs.append(msg)

    for name in BROWSE_ONLY + QUIET:
        def _rec(*a, _n=name, **k):
            calls.append(_n)
        setattr(ctrl, name, _rec)
    return ctrl


class TestAdoptRunsOnlyTheQuietHalf:
    def test_adopt_performs_every_quiet_item(self):
        calls = []
        ctrl = _controller(calls)
        ctrl.adopt_workspace("C:/ws/new")
        for name in QUIET:
            assert name in calls, f"adopt skipped the quiet item {name}"

    def test_adopt_performs_no_browse_only_item(self):
        calls = []
        ctrl = _controller(calls)
        ctrl.adopt_workspace("C:/ws/new")
        leaked = [n for n in BROWSE_ONLY if n in calls]
        assert leaked == [], f"adopt reached Browse-only work: {leaked}"

    def test_adopt_fills_the_path_box_and_working_dir(self):
        ctrl = _controller([])
        ctrl.adopt_workspace("C:/ws/new")
        assert ctrl.working_dir == "C:/ws/new"
        assert ctrl.ui.set_workdir_path.text() == "C:/ws/new"

    def test_adopt_clears_the_previous_context(self):
        ctrl = _controller([])
        ctrl.adopt_workspace("C:/ws/new")
        assert ctrl.loaded_stack_path == ""
        assert ctrl.loaded_patch_name == ""
        assert ctrl.sample_records == {}
        assert ctrl.active_run_id == "", (
            "active_run_id must be blank so _load_internal_state's peek "
            "resolves the NEW workspace's run")
        assert ctrl.ui.loaded_patch_label.text() == ""

    def test_adopt_runs_under_the_busy_bar(self):
        ctrl = _controller([])
        ctrl.adopt_workspace("C:/ws/new")
        assert (0, 0) in ctrl.ui.patch_busy_bar.ranges, "no scan-mode busy bar"
        assert ctrl._busy_depth == 0, "busy scope leaked"

    def test_busy_scope_is_released_even_when_the_load_raises(self):
        ctrl = _controller([])

        def _boom():
            raise RuntimeError("corrupt state")
        ctrl._load_internal_state = _boom
        with pytest.raises(RuntimeError):
            ctrl.adopt_workspace("C:/ws/new")
        assert ctrl._busy_depth == 0

    def test_empty_path_is_ignored(self):
        calls = []
        ctrl = _controller(calls)
        ctrl.adopt_workspace("")
        assert calls == []
        assert ctrl.working_dir == ""


class TestAdoptNeverOverwritesABrowsedPath:
    """Invariant I2 — an explicit choice is never taken away."""

    def test_no_op_when_the_step_already_has_a_workspace(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/mine")
        ctrl.ui.set_workdir_path.setText("C:/ws/mine")
        ctrl.adopt_workspace("C:/ws/other")
        assert calls == []
        assert ctrl.working_dir == "C:/ws/mine"

    def test_no_op_when_previous_is_not_what_this_step_holds(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/mine")
        ctrl.adopt_workspace("C:/ws/other", previous="C:/ws/somewhere_else")
        assert calls == []
        assert ctrl.working_dir == "C:/ws/mine"

    def test_re_adopts_the_path_it_inherited(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        assert ctrl.working_dir == "C:/ws/B"
        for name in QUIET:
            assert name in calls

    def test_previous_matches_case_and_slash_insensitively(self):
        ctrl = _controller([], working_dir="C:/ws/A")
        ctrl.adopt_workspace("C:/ws/B", previous="c:\\WS\\a")
        assert ctrl.working_dir == "C:/ws/B"

    def test_re_adopt_to_the_same_folder_is_a_no_op(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl.adopt_workspace("C:/ws/A", previous="C:/ws/A")
        assert calls == []


class TestReAdoptTearsDownTheOldProject:
    """B2 item 4 — the shared workspace moving out from under Step 2.

    Adopt on a VIRGIN step touches no layers (DOES NOT CHANGE).  A
    RE-adopt is a workspace switch, so it performs the same teardown Browse
    does — otherwise project A's annotator keeps driving project B's data,
    the exact bug `fix/step2-workspace-switch-channels` fixed for Browse.
    """

    def test_re_adopt_tears_down_and_drops_the_segmenter(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        assert "_close_step2_annotator_dock_safely" in calls
        assert "_clear_patch_viewer_layers_safely" in calls
        assert "_flush_pending_state_timers" in calls
        assert ctrl.segmentation_manager.segmenter is None

    def test_teardown_runs_before_the_context_is_cleared(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        assert (calls.index("_clear_patch_viewer_layers_safely")
                < calls.index("_load_internal_state"))

    def test_a_virgin_adopt_touches_no_layers(self):
        calls = []
        ctrl = _controller(calls, working_dir="")
        ctrl.adopt_workspace("C:/ws/B")
        assert "_close_step2_annotator_dock_safely" not in calls
        assert "_clear_patch_viewer_layers_safely" not in calls
        assert ctrl.segmentation_manager.segmenter is not None

    def test_re_adopt_still_shows_no_modal_and_publishes_nothing(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        for name in ("_maybe_prompt_step2_migration",
                     "_maybe_prompt_split_to_flat_migration",
                     "_maybe_prompt_first_run",
                     "_auto_reload_saved_patch",
                     "_publish_workspace_to_shell"):
            assert name not in calls, f"re-adopt reached {name}"


class TestPublishIsSuppressedDuringAdopt:
    def test_the_flag_is_set_while_the_box_is_filled_and_cleared_after(self):
        seen = []
        ctrl = _controller([])
        line = ctrl.ui.set_workdir_path
        real = line.setText
        line.setText = lambda v: (seen.append(ctrl._suppress_publish), real(v))
        ctrl.adopt_workspace("C:/ws/new")
        assert seen == [True], f"publish was not suppressed: {seen}"
        assert ctrl._suppress_publish is False

    def test_publish_is_a_no_op_while_suppressed(self):
        published = []

        class _Shell:
            def set_workspace(self, p, source_step=""):
                published.append((p, source_step))

        ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
        ctrl._workflow_shell = _Shell()
        ctrl._suppress_publish = True
        ctrl._publish_workspace_to_shell("C:/ws/new")
        assert published == []
        # Teeth: without the flag it WOULD publish, so the assertion above is
        # about the flag and not about a dead code path.
        ctrl._suppress_publish = False
        ctrl._publish_workspace_to_shell("C:/ws/new")
        assert published == [("C:/ws/new", "roi_hitl_2d")]


class TestReEntryRefresh:
    """A patch extracted or trimmed in Step 1 shows up on entry."""

    def test_entry_refreshes_the_lists_when_a_workspace_is_set(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl._populate_patch_list_from_disk = lambda **k: calls.append(
            ("populate", k))
        ctrl.on_step_enter()
        assert ("populate", {"notify": False}) in calls
        assert "_refresh_patch_list_badges" in calls

    def test_entry_loads_no_state_and_no_patch(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl._populate_patch_list_from_disk = lambda **k: None
        ctrl.on_step_enter()
        assert "_load_internal_state" not in calls
        assert "_auto_reload_saved_patch" not in calls

    def test_entry_is_a_no_op_without_a_workspace(self):
        calls = []
        ctrl = _controller(calls, working_dir="")
        ctrl._populate_patch_list_from_disk = lambda **k: calls.append("populate")
        ctrl.on_step_enter()
        assert calls == []

    def test_entry_clears_the_left_flag(self):
        ctrl = _controller([], working_dir="")
        ctrl._step2_left = True
        ctrl.on_step_enter()
        assert ctrl._step2_left is False

    def test_a_failing_refresh_never_blocks_the_step_switch(self):
        ctrl = _controller([], working_dir="C:/ws/A")

        def _boom(**k):
            raise OSError("drive gone")
        ctrl._populate_patch_list_from_disk = _boom
        ctrl.on_step_enter()          # must not raise
        assert ctrl._step2_left is False


# ---------------------------------------------------------------------------
# Second B2 audit, FIX-FIRST #2 — a re-adopt must not run under a trainer
# ---------------------------------------------------------------------------

class _Thread:
    def __init__(self, running):
        self._running = running

    def isRunning(self):
        return self._running


class TestReAdoptIsBlockedWhileTraining:
    """`_switch_run_guard.check_switch_safety` HARD-BLOCKS a run switch while
    training runs.  A re-adopt is the same kind of event — it tears down the
    annotator and loads another workspace — but the shell drives it without
    asking anyone.  It must decline, leaving Step 2 where it is (the shell
    then shows it as diverged/amber)."""

    def test_a_running_training_thread_blocks_the_re_adopt(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl._training_thread = _Thread(True)
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        assert ctrl.working_dir == "C:/ws/A"
        assert calls == [], f"work happened under a live trainer: {calls}"
        assert ctrl.segmentation_manager.segmenter is not None

    def test_a_synchronous_training_run_blocks_it_too(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl._sync_training_in_flight = True
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        assert ctrl.working_dir == "C:/ws/A"
        assert calls == []

    def test_the_user_is_told_in_plain_words(self):
        ctrl = _controller([], working_dir="C:/ws/A")
        ctrl._training_thread = _Thread(True)
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        assert any("training" in m.lower() for m in ctrl.logs), ctrl.logs

    def test_a_finished_thread_does_not_block(self):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl._training_thread = _Thread(False)
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        assert ctrl.working_dir == "C:/ws/B"

    def test_an_indeterminate_thread_is_treated_as_running(self):
        class _Boom:
            def isRunning(self):
                raise RuntimeError("gone")
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl._training_thread = _Boom()
        ctrl.adopt_workspace("C:/ws/B", previous="C:/ws/A")
        assert ctrl.working_dir == "C:/ws/A", (
            "indeterminate means blocked — the same call the run-switch "
            "guard makes")

    def test_a_virgin_adopt_is_not_blocked(self):
        """Nothing is torn down and no trainer can be running without a
        workspace, so a first adopt is never in the trainer's way."""
        calls = []
        ctrl = _controller(calls, working_dir="")
        ctrl._training_thread = _Thread(True)
        ctrl.adopt_workspace("C:/ws/B")
        assert ctrl.working_dir == "C:/ws/B"


class TestReAdoptTellsTheUserWhatHappened:
    """Journey note (B): a re-adopt closes the previous project's patch and
    annotator, and until now the only trace was a console line."""

    def test_a_re_adopt_logs_a_plain_line(self):
        ctrl = _controller([], working_dir="C:/ws/A")
        ctrl.adopt_workspace("C:/ws/ProjectB", previous="C:/ws/A")
        joined = " ".join(ctrl.logs)
        assert "ProjectB" in joined
        assert "closed" in joined.lower(), ctrl.logs

    def test_a_virgin_adopt_does_not_claim_anything_was_closed(self):
        ctrl = _controller([], working_dir="")
        ctrl.adopt_workspace("C:/ws/ProjectB")
        assert not any("closed" in m.lower() for m in ctrl.logs), ctrl.logs

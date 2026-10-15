"""Step 3 `adopt_workspace` — the shared-workspace hand-over.

Adopt runs the quiet half of Browse: caches reset, state load, stack-dir
restore, lists + model widgets + run picker.  It does NOT run the layout
migration modal, and — the point of it — it is **read-only** for a
workspace whose `step3_state.json` predates the v3 runs envelope: `load`
upgrades v1/v2 → v3 in memory and the next save would rewrite the file with
no backup, so walking into the step must not convert a file nobody asked to
touch.

Two other B2 items live here:

* invariant **I4** — `state_manager.active_run_id` is blanked before the
  load on BOTH paths.  This is genuinely new for Browse too: no such
  assignment existed anywhere in the handler, so a workspace switch kept the
  previous workspace's run.
* **** — the cheap list refresh on every entry.

Qt-free: the controller is built with `__new__` and stub collaborators.
"""
from __future__ import annotations

import json
import os

import pytest

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)
from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)

#: Browse does these; adopt must not.
BROWSE_ONLY = ("_prompt_step3_migration", "_publish_workspace_to_shell")

#: The quiet half, which both paths perform.
QUIET = (
    "_reset_workspace_scoped_caches",
    "_refresh_patch_state_list",
    "_clear_model_widgets",
    "_refresh_patch_list_colors",
    "_refresh_summary_dashboard",
    "_populate_refinement_source_combo",
    "_refresh_refinement_patch_list",
    "_refresh_per_orientation_summary",
    "_refresh_mode_hint",
    "_refresh_run_picker",
)


class _Item:
    def __init__(self, name):
        self._name = name
        self._selected = False

    def text(self):
        return self._name

    def data(self, role):
        return self._name

    def setSelected(self, v):
        self._selected = v

    def isSelected(self):
        return self._selected


class _PatchList:
    """Just enough QListWidget for the re-entry refresh."""

    def __init__(self, names=()):
        self._items = [_Item(n) for n in names]
        self.clears = 0

    def count(self):
        return len(self._items)

    def item(self, i):
        return self._items[i]

    def clear(self):
        self.clears += 1
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def selectedItems(self):
        return [i for i in self._items if i.isSelected()]

    def setCurrentItem(self, item):
        for i in self._items:
            i.setSelected(i is item)

    def select(self, name):
        for i in self._items:
            i.setSelected(i.text() == name)


class _Line:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value


class _Check:
    def __init__(self):
        self._checked = False
        self.blocked = []

    def blockSignals(self, on):
        self.blocked.append(on)

    def setChecked(self, v):
        self._checked = v

    def isChecked(self):
        return self._checked


class _UI:
    def __init__(self):
        self.set_workdir_path = _Line()
        self.downsample_mode_cb = _Check()
        # The real UI always has this; the re-entry refresh reads it to
        # decide whether the patch folder actually changed.
        self.loaded_patch_list = _PatchList()


def _controller(calls, tmp_path=None, working_dir=""):
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.ui = _UI()
    ctrl.working_dir = working_dir
    ctrl.loaded_stack_dir = ""
    ctrl.state_manager = ReconstructionStateManager()
    ctrl._suppress_publish = False
    ctrl._workflow_shell = None
    ctrl.logs = []
    ctrl._append_log = lambda msg: ctrl.logs.append(msg)
    ctrl._restore_persisted_model_selection = lambda: (
        calls.append("_restore_persisted_model_selection") or False)
    ctrl._discover_workspace_models = lambda: calls.append(
        "_discover_workspace_models")
    ctrl._load_stack_dir = lambda d, *, persist=True: (
        calls.append(("_load_stack_dir", d)),
        setattr(ctrl, "loaded_stack_dir", d))
    for name in BROWSE_ONLY + QUIET:
        def _rec(*a, _n=name, **k):
            calls.append(_n)
        setattr(ctrl, name, _rec)
    return ctrl


def _write_state(ws, payload):
    (ws / "step3_state.json").write_text(json.dumps(payload), encoding="utf-8")


class TestAdoptRunsTheQuietHalf:
    def test_every_quiet_item_runs(self, tmp_path):
        calls = []
        ctrl = _controller(calls)
        ctrl.adopt_workspace(str(tmp_path))
        for name in QUIET:
            assert name in calls, f"adopt skipped {name}"
        assert ("_restore_persisted_model_selection" in calls
                and "_discover_workspace_models" in calls), (
            "model restore falls back to discovery when nothing is persisted")

    def test_no_browse_only_item_runs(self, tmp_path):
        calls = []
        ctrl = _controller(calls)
        ctrl.adopt_workspace(str(tmp_path))
        leaked = [n for n in BROWSE_ONLY if n in calls]
        assert leaked == [], f"adopt reached Browse-only work: {leaked}"

    def test_the_path_box_and_working_dir_are_filled(self, tmp_path):
        ctrl = _controller([])
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.working_dir == str(tmp_path)
        assert ctrl.ui.set_workdir_path.text() == str(tmp_path)

    def test_state_is_loaded_from_the_workspace(self, tmp_path):
        _write_state(tmp_path, {"schema_version": 3, "runs": {
            "default": {"patches": {"p1.tif": {"status": "reconstructed"}}}}})
        ctrl = _controller([])
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.state_manager.get_patch_status("p1.tif") == "reconstructed"

    def test_the_saved_stack_dir_is_restored(self, tmp_path):
        stacks = tmp_path / "patches"
        stacks.mkdir()
        _write_state(tmp_path, {"schema_version": 3, "runs": {"default": {
            "patches": {}, "loaded_stack_dir": str(stacks)}}})
        calls = []
        ctrl = _controller(calls)
        ctrl.adopt_workspace(str(tmp_path))
        assert ("_load_stack_dir", str(stacks)) in calls

    def test_downsample_toggle_is_synced_with_signals_blocked(self, tmp_path):
        ctrl = _controller([])
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.ui.downsample_mode_cb.blocked == [True, False]

    def test_empty_path_is_ignored(self, tmp_path):
        calls = []
        ctrl = _controller(calls)
        ctrl.adopt_workspace("")
        assert calls == []


class TestAdoptNeverOverwritesABrowsedPath:
    def test_no_op_when_the_step_already_has_a_workspace(self, tmp_path):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/mine")
        ctrl.adopt_workspace(str(tmp_path))
        assert calls == []
        assert ctrl.working_dir == "C:/ws/mine"

    def test_re_adopts_the_path_it_inherited(self, tmp_path):
        calls = []
        ctrl = _controller(calls, working_dir="C:/ws/A")
        ctrl.adopt_workspace(str(tmp_path), previous="C:/ws/A")
        assert ctrl.working_dir == str(tmp_path)
        assert "_reset_workspace_scoped_caches" in calls

    def test_re_adopt_to_the_same_folder_is_a_no_op(self, tmp_path):
        calls = []
        ctrl = _controller(calls, working_dir=str(tmp_path))
        ctrl.adopt_workspace(str(tmp_path), previous=str(tmp_path))
        assert calls == []


class TestActiveRunIdIsResetOnBothPaths:
    """Invariant I4 — new behaviour for Browse as well as for adopt."""

    def test_adopt_resolves_the_run_from_the_new_workspace(self, tmp_path):
        _write_state(tmp_path, {
            "schema_version": 3,
            "active_run_id": "cells",
            "runs": {"default": {"patches": {}},
                     "cells": {"patches": {"c.tif": {"status": "predicted"}}}},
        })
        ctrl = _controller([])
        ctrl.state_manager.active_run_id = "leftover_from_project_A"
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.state_manager.active_run_id == "cells"
        assert ctrl.state_manager.get_patch_status("c.tif") == "predicted"

    def test_the_quiet_half_blanks_it_before_the_load(self, tmp_path):
        ctrl = _controller([])
        ctrl.state_manager.active_run_id = "leftover"
        seen = {}
        real_load = ctrl.state_manager.load

        def _spy(ws):
            seen["run_at_load"] = ctrl.state_manager.active_run_id
            return real_load(ws)
        ctrl.state_manager.load = _spy
        ctrl._adopt_workspace_quiet(str(tmp_path))
        assert seen["run_at_load"] == "", (
            "active_run_id must be blank when load runs, or the PREVIOUS "
            "workspace's run decides which run of the new file is unwrapped")

    def test_browse_shares_the_same_quiet_half(self):
        import inspect
        src = inspect.getsource(FusionReconstructionController._browse_workdir)
        assert "_adopt_workspace_quiet(" in src, (
            "Browse must run the same quiet half, so I4's reset applies to "
            "both paths and cannot drift apart")


class TestPreV3IsReadOnly:
    """Adopt reads an old state file and writes nothing."""

    def _pre_v3(self, tmp_path):
        _write_state(tmp_path, {
            "schema_version": 2,
            "patches": {"old.tif": {"status": "reconstructed"}},
        })
        return (tmp_path / "step3_state.json").read_bytes()

    def test_the_file_is_byte_identical_after_adopt(self, tmp_path):
        before = self._pre_v3(tmp_path)
        ctrl = _controller([])
        ctrl.adopt_workspace(str(tmp_path))
        assert (tmp_path / "step3_state.json").read_bytes() == before

    def test_the_state_is_still_usable_in_memory(self, tmp_path):
        self._pre_v3(tmp_path)
        ctrl = _controller([])
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.state_manager.get_patch_status("old.tif") == "reconstructed"

    def test_a_save_scheduled_during_adopt_cannot_land_afterwards(self, tmp_path):
        before = self._pre_v3(tmp_path)
        ctrl = _controller([])
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.state_manager.suppress_save is False, (
            "the flag is scoped to adopt; a later explicit user action must "
            "still be able to save")
        assert (tmp_path / "step3_state.json").read_bytes() == before

    def test_suppression_is_lifted_even_when_the_load_raises(self, tmp_path):
        self._pre_v3(tmp_path)
        ctrl = _controller([])

        def _boom():
            raise RuntimeError("list rebuild failed")
        ctrl._refresh_patch_state_list = _boom
        with pytest.raises(RuntimeError):
            ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.state_manager.suppress_save is False

    def test_a_pending_save_from_the_previous_workspace_is_flushed_first(
            self, tmp_path):
        """Found by the B2 live run: a debounced save armed by workspace A
        resolves its destination LAZILY, at delivery time.  Cross into
        workspace B inside that 500 ms window and A's state lands in B's
        file — it rewrote the very pre-v3 file adopt had left alone.  The
        flush must therefore happen while `_workspace` still points at A."""
        old_ws = tmp_path / "A"
        old_ws.mkdir()
        new_ws = tmp_path / "B"
        new_ws.mkdir()
        _write_state(new_ws, {"schema_version": 2, "patches": {}})
        untouched = (new_ws / "step3_state.json").read_bytes()

        ctrl = _controller([])
        ctrl.state_manager.load(str(old_ws))
        ctrl.state_manager.set_patch_status("a.tif", "predicted")

        # The real `flush` is a no-op without a live Qt event loop, so the
        # pending save is emulated: whatever `flush` is called on must still
        # be pointed at workspace A, and writing then must land in A.
        flushed_at = {}

        def _spy():
            flushed_at["workspace"] = ctrl.state_manager._workspace
            ctrl.state_manager.save()
        ctrl.state_manager.flush = _spy

        ctrl.adopt_workspace(str(new_ws), previous=str(old_ws))
        assert flushed_at.get("workspace") == str(old_ws), (
            "the flush must run BEFORE load repoints the state manager")
        assert (old_ws / "step3_state.json").is_file(), (
            "workspace A's pending state must land in workspace A")
        assert (new_ws / "step3_state.json").read_bytes() == untouched

    def test_a_v3_workspace_is_not_suppressed(self, tmp_path):
        _write_state(tmp_path, {"schema_version": 3,
                                "runs": {"default": {"patches": {}}}})
        ctrl = _controller([])
        seen = []
        real = ctrl.state_manager.load
        ctrl.state_manager.load = lambda ws: (
            seen.append(ctrl.state_manager.suppress_save), real(ws))[1]
        ctrl.adopt_workspace(str(tmp_path))
        assert seen == [False]

    def test_a_fresh_workspace_is_not_suppressed(self, tmp_path):
        ctrl = _controller([])
        seen = []
        real = ctrl.state_manager.load
        ctrl.state_manager.load = lambda ws: (
            seen.append(ctrl.state_manager.suppress_save), real(ws))[1]
        ctrl.adopt_workspace(str(tmp_path))
        assert seen == [False]


class TestReEntryRefresh:
    """Patches extracted in Step 1 appear without a Browse."""

    def test_entry_refreshes_the_patch_state_list(self, tmp_path):
        calls = []
        ctrl = _controller(calls, working_dir=str(tmp_path))
        ctrl._refresh_lists_on_step_enter()
        assert "_refresh_patch_state_list" in calls

    def test_entry_rescans_a_loaded_stack_dir(self, tmp_path):
        stacks = tmp_path / "patches"
        stacks.mkdir()
        # A patch that is on disk but not in the list — i.e. one Step 1 just
        # extracted.  (An UNCHANGED directory must NOT be re-read; that is
        # covered by TestReEntryRefreshIsCheapAndKeepsSelection.)
        (stacks / "fresh.tif").write_bytes(b"")
        calls = []
        ctrl = _controller(calls, working_dir=str(tmp_path))
        ctrl.loaded_stack_dir = str(stacks)
        ctrl._refresh_lists_on_step_enter()
        assert ("_load_stack_dir", str(stacks)) in calls, (
            "a patch renamed or extracted in Step 1 only shows up if the "
            "directory is re-read")

    def test_a_vanished_stack_dir_is_skipped_not_crashed(self, tmp_path):
        calls = []
        ctrl = _controller(calls, working_dir=str(tmp_path))
        ctrl.loaded_stack_dir = str(tmp_path / "gone")
        ctrl._refresh_lists_on_step_enter()
        assert not any(isinstance(c, tuple) for c in calls)
        assert "_refresh_patch_state_list" in calls

    def test_entry_is_a_no_op_without_a_workspace(self):
        calls = []
        ctrl = _controller(calls, working_dir="")
        ctrl._refresh_lists_on_step_enter()
        assert calls == []

    def test_entry_writes_nothing_on_a_pre_v3_workspace(self, tmp_path):
        _write_state(tmp_path, {"schema_version": 2, "patches": {}})
        before = (tmp_path / "step3_state.json").read_bytes()
        stacks = tmp_path / "patches"
        stacks.mkdir()
        (stacks / "fresh.tif").write_bytes(b"")   # forces the reload path
        ctrl = _controller([], working_dir=str(tmp_path))
        ctrl.state_manager.load(str(tmp_path))
        ctrl.loaded_stack_dir = str(stacks)
        # The real `_load_stack_dir` registers patches, which ends in
        # schedule_save; a flush landing inside the refresh must still not
        # rewrite a pre-v3 file.
        def _register_and_flush(d):
            ctrl.state_manager.register_patches(["a.tif"])
            ctrl.state_manager.save()
        ctrl._load_stack_dir = _register_and_flush
        ctrl._refresh_lists_on_step_enter()
        assert (tmp_path / "step3_state.json").read_bytes() == before
        # Teeth: the suppression is scoped to the refresh, so an explicit
        # user save afterwards still works (and would have failed the
        # assertion above had it run inside).
        ctrl.state_manager.save()
        assert (tmp_path / "step3_state.json").read_bytes() != before

    def test_a_failing_refresh_never_blocks_the_step_switch(self, tmp_path):
        ctrl = _controller([], working_dir=str(tmp_path))

        def _boom():
            raise OSError("drive gone")
        ctrl._refresh_patch_state_list = _boom
        ctrl._refresh_lists_on_step_enter()      # must not raise

    def test_on_step_enter_calls_the_refresh(self):
        import inspect
        src = inspect.getsource(FusionReconstructionController.on_step_enter)
        assert "_refresh_lists_on_step_enter()" in src


# ---------------------------------------------------------------------------
# B2 lifecycle audit, FIX-FIRST #1 — the re-entry refresh must be cheap AND
# must not throw away the user's patch selection.
# ---------------------------------------------------------------------------

def _reentry_controller(tmp_path, on_disk, listed, selected=""):
    stacks = tmp_path / "patches"
    stacks.mkdir(exist_ok=True)
    for n in on_disk:
        (stacks / n).write_bytes(b"")
    calls = []
    ctrl = _controller(calls, working_dir=str(tmp_path))
    ctrl.loaded_stack_dir = str(stacks)
    ctrl.ui.loaded_patch_list = _PatchList(listed)
    if selected:
        ctrl.ui.loaded_patch_list.select(selected)

    def _reload(d, *, persist=True):
        calls.append(("_load_stack_dir", d))
        lst = ctrl.ui.loaded_patch_list
        lst.clear()
        for n in sorted(os.listdir(d)):
            lst.addItem(_Item(n))
    ctrl._load_stack_dir = _reload
    return ctrl, calls, str(stacks)


class TestReEntryRefreshIsCheapAndKeepsSelection:
    def test_unchanged_directory_does_not_reload(self, tmp_path):
        names = ["a.tif", "b.tif"]
        ctrl, calls, _ = _reentry_controller(tmp_path, names, names)
        ctrl._refresh_lists_on_step_enter()
        assert not any(isinstance(c, tuple) for c in calls), (
            "nothing changed on disk — re-reading the directory (and opening "
            "a TiffFile per patch) on every step switch is not free")
        assert ctrl.ui.loaded_patch_list.clears == 0
        assert "_refresh_patch_state_list" in calls, (
            "adopt still asks for the state-list refresh on every entry")

    def test_unchanged_directory_keeps_the_selection(self, tmp_path):
        names = ["a.tif", "b.tif"]
        ctrl, _, _ = _reentry_controller(tmp_path, names, names, selected="b.tif")
        ctrl._refresh_lists_on_step_enter()
        assert ctrl._selected_patch_name() == "b.tif"

    def test_a_new_patch_triggers_a_reload(self, tmp_path):
        ctrl, calls, stacks = _reentry_controller(
            tmp_path, ["a.tif", "b.tif"], ["a.tif"])
        ctrl._refresh_lists_on_step_enter()
        assert ("_load_stack_dir", stacks) in calls

    def test_a_renamed_patch_triggers_a_reload(self, tmp_path):
        ctrl, calls, stacks = _reentry_controller(
            tmp_path, ["a.tif", "b_RENAMED.tif"], ["a.tif", "b.tif"])
        ctrl._refresh_lists_on_step_enter()
        assert ("_load_stack_dir", stacks) in calls
        listed = [ctrl.ui.loaded_patch_list.item(i).text()
                  for i in range(ctrl.ui.loaded_patch_list.count())]
        assert listed == ["a.tif", "b_RENAMED.tif"]

    def test_a_reload_restores_a_still_present_selection(self, tmp_path):
        ctrl, _, _ = _reentry_controller(
            tmp_path, ["a.tif", "b.tif", "c.tif"], ["a.tif", "b.tif"],
            selected="b.tif")
        ctrl._refresh_lists_on_step_enter()
        assert ctrl._selected_patch_name() == "b.tif", (
            "a patch that is still on disk must survive the rebuild")

    def test_a_selection_that_vanished_is_simply_dropped(self, tmp_path):
        ctrl, _, _ = _reentry_controller(
            tmp_path, ["a.tif"], ["a.tif", "b.tif"], selected="b.tif")
        ctrl._refresh_lists_on_step_enter()
        assert ctrl._selected_patch_name() == ""

    def test_only_stack_suffixes_count_as_a_change(self, tmp_path):
        names = ["a.tif", "b.tif"]
        ctrl, calls, stacks = _reentry_controller(tmp_path, names, names)
        (tmp_path / "patches" / "a_metadata.json").write_text("{}",
                                                             encoding="utf-8")
        ctrl._refresh_lists_on_step_enter()
        assert not any(isinstance(c, tuple) for c in calls), (
            "the sidecar JSON Step 1 writes next to every patch must not "
            "look like a new patch")


class TestAxesDetectionIsNotRepeated:
    """`_load_stack_dir` opened a TiffFile per patch, every time."""

    def _ctrl(self, tmp_path, cached):
        stacks = tmp_path / "patches"
        stacks.mkdir()
        for n in ("a.tif", "b.tif"):
            (stacks / n).write_bytes(b"")
        calls = []
        ctrl = _controller(calls, working_dir=str(tmp_path))
        del ctrl._load_stack_dir            # exercise the real one
        ctrl.ui.loaded_patch_list = _PatchList()
        ctrl.ui.set_patch_directory_path = _Line()
        ctrl._warn_patch_run_slug_collisions = lambda names: None
        ctrl.state_manager.load(str(tmp_path))
        ctrl.state_manager.register_patches(["a.tif", "b.tif"])
        for name in cached:
            ctrl.state_manager.update_patch_metadata(name, axes="ZYX")
        detected = []
        ctrl._detect_and_cache_patch_axes = lambda n: detected.append(n)
        return ctrl, detected, str(stacks)

    def test_cached_patches_are_not_re_read(self, tmp_path):
        ctrl, detected, stacks = self._ctrl(tmp_path, cached=("a.tif", "b.tif"))
        ctrl._load_stack_dir(stacks)
        assert detected == [], (
            "a patch whose axes are already recorded must not have its TIFF "
            "header opened again — that is a file open per patch per entry")

    def test_uncached_patches_are_still_detected(self, tmp_path):
        ctrl, detected, stacks = self._ctrl(tmp_path, cached=("a.tif",))
        ctrl._load_stack_dir(stacks)
        assert detected == ["b.tif"]

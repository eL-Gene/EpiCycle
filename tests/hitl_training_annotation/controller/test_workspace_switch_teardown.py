"""Regression tests for the workspace-switch annotator/layer teardown fix
(``fix/step2-workspace-switch-channels``, 2026-06-03).

Bug: switching workspaces via Browse (``_browse_workdir``) with the 2D
annotator left OPEN kept the old annotator mounted, driving the newly-loaded
workspace with stale layer wiring + channel context.  A segment in the new
workspace then produced a mask that never displayed.  Root cause: unlike the
run-switch path (``_switch_active_run``), ``_browse_workdir`` never closed the
annotator dock or cleared the stale patch layers.

Fix: ``_browse_workdir`` now calls the SAME teardown helpers
``_switch_active_run`` uses — ``_close_step2_annotator_dock_safely`` +
``_clear_patch_viewer_layers_safely`` — before loading the new workspace.

These are source-inspection tests (``_browse_workdir`` opens a QFileDialog and
drives the live viewer, so it can't be unit-driven without Qt) — same approach
as test_label_compaction.py's wiring assertions.
"""

from __future__ import annotations

import inspect

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


def test_browse_workdir_tears_down_annotator_and_layers():
    """The workspace-switch path must close the stale annotator dock AND clear
    stale patch layers -- the two-call teardown that fixes the cross-workspace
    stale-wiring bug.

    B2 (shared workspace) moved the three teardown blocks out of
    ``_browse_workdir`` into ``_teardown_previous_workspace``, so that a
    RE-ADOPT (the shared workspace moving while Step 2 holds another project)
    performs exactly the same teardown instead of a parallel one.  These
    assertions follow the code: the helper still does the work, and Browse
    still calls it first."""
    src = inspect.getsource(
        TrainingAnnotationController._teardown_previous_workspace)
    assert "_close_step2_annotator_dock_safely()" in src, (
        "the workspace-switch teardown must close the open annotator -- "
        "leaving it mounted drives the new workspace with stale wiring."
    )
    assert "_clear_patch_viewer_layers_safely()" in src, (
        "the workspace-switch teardown must clear the previous workspace's "
        "stale viewer layers."
    )
    # The two calls must stay in this order: the dock close hands the layers
    # back before the sweep removes them.
    assert (src.index("_close_step2_annotator_dock_safely()")
            < src.index("_clear_patch_viewer_layers_safely()")), (
        "the annotator dock must close BEFORE its patch layers are removed"
    )
    browse = inspect.getsource(TrainingAnnotationController._browse_workdir)
    assert "_teardown_previous_workspace()" in browse, (
        "Browse must still run the teardown before loading the new workspace."
    )


def test_browse_workdir_teardown_runs_before_context_is_cleared():
    """The teardown helpers identify layers via the OLD ``loaded_patch_name``,
    so they MUST run before the in-memory scalar fields (incl.
    loaded_patch_name) are blanked -- otherwise collect_patch_layer_names
    finds nothing and the stale layers leak.

    Post-B2 the blanking lives in ``_load_workspace_quiet``, so the ordering
    to lock is teardown-before-quiet-load within ``_browse_workdir`` (plus the
    fact that the blanking really is in that second half)."""
    src = inspect.getsource(TrainingAnnotationController._browse_workdir)
    teardown_idx = src.index("_teardown_previous_workspace()")
    load_idx = src.index("_load_workspace_quiet()")
    assert teardown_idx < load_idx, (
        "annotator/layer teardown must run BEFORE the quiet load blanks "
        "loaded_patch_name (teardown needs the old patch name)."
    )
    quiet = inspect.getsource(TrainingAnnotationController._load_workspace_quiet)
    assert 'self.loaded_patch_name = ""' in quiet, (
        "the context blanking must stay in the quiet half -- if it moves, the "
        "ordering assertion above stops meaning anything."
    )


def test_browse_workdir_resets_stale_segmenter():
    """The workspace switch must drop the previous workspace's live segmenter
    so the next segment rebuilds from the new workspace's restored config.
    Without this, the segmenter-rebuild guard (config-vs-config) is blind to
    the stale segmenter object and runs the WRONG model on the new data --
    the user-reported 'mask doesn't appear after switch' bug."""
    src = inspect.getsource(
        TrainingAnnotationController._teardown_previous_workspace)
    assert "segmenter = None" in src, (
        "the workspace-switch teardown must null segmentation_manager."
        "segmenter so the next segment rebuilds from the new config."
    )


def test_run_segmentation_rebuilds_segmenter_when_none():
    """The segmenter-null fix RELIES on _run_segmentation rebuilding the
    segmenter when it's None.  Lock that premise: if someone changes the
    rebuild guard, this catches that the fix's mechanism broke (nulling would
    no longer force a rebuild from the new workspace's config)."""
    src = inspect.getsource(TrainingAnnotationController._run_segmentation)
    assert "self.segmentation_manager.segmenter is None" in src, (
        "_run_segmentation must rebuild the segmenter when None -- this is "
        "what makes the workspace-switch segmenter-null fix take effect."
    )
    assert "set_model_preference(self.current_config" in src, (
        "the rebuild must use self.current_config (the freshly-restored new-"
        "workspace config), not a stale snapshot."
    )


def test_browse_workdir_segmenter_reset_runs_before_state_load():
    """The segmenter reset must happen before _load_internal_state replaces
    current_config -- though either order works for the null itself, doing it
    in the teardown block keeps it grouped with the other stale-state drops."""
    src = inspect.getsource(TrainingAnnotationController._browse_workdir)
    reset_idx = src.index("_teardown_previous_workspace()")
    load_idx = src.index("_load_workspace_quiet()")
    assert reset_idx < load_idx
    quiet = inspect.getsource(TrainingAnnotationController._load_workspace_quiet)
    assert "_load_internal_state()" in quiet


def test_browse_workdir_teardown_runs_after_timer_flush():
    """Teardown must run after the pending-timer flush (so the old workspace's
    state is persisted before its annotator is torn down), matching the
    _switch_active_run ordering (flush -> close -> clear)."""
    src = inspect.getsource(
        TrainingAnnotationController._teardown_previous_workspace)
    flush_idx = src.index("_flush_pending_state_timers()")
    close_idx = src.index("_close_step2_annotator_dock_safely()")
    assert flush_idx < close_idx, (
        "timer flush (persisting old workspace state) must precede annotator "
        "teardown."
    )


def test_teardown_helpers_exist_and_are_callable():
    """Sanity: the borrowed helpers exist on the controller (we reuse the
    run-switch teardown rather than inventing a parallel path)."""
    assert hasattr(TrainingAnnotationController, "_close_step2_annotator_dock_safely")
    assert hasattr(TrainingAnnotationController, "_clear_patch_viewer_layers_safely")


# ---------------------------------------------------------------------------
# Behavioral test of the layer-clear helper (audit follow-up): the source-
# inspection tests above never RUN the removal logic.  This drives it against
# a fake viewer to prove it removes the previous workspace's patch layers,
# spares unrelated layers, and hides-before-remove (vispy rule).
# ---------------------------------------------------------------------------

class _FakeLayer:
    def __init__(self, name):
        self.name = name
        self.visible = True


class _FakeLayers(list):
    """Mimics napari viewer.layers: `in`/`[]` by name + remove(name)."""

    def __contains__(self, item):
        if isinstance(item, str):
            return any(l.name == item for l in self)
        return list.__contains__(self, item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if l.name == key:
                    return l
            raise KeyError(key)
        return list.__getitem__(self, key)

    def remove(self, item):
        if isinstance(item, str):
            for l in list(self):
                if l.name == item:
                    # hide-before-remove invariant: the helper MUST have set
                    # visible=False before removing (vispy rule).
                    assert l.visible is False, (
                        f"layer {item!r} removed while still visible — "
                        "violates hide-before-remove"
                    )
                    list.remove(self, l)
                    return
            raise ValueError(item)
        return list.remove(self, item)


class _FakeViewer:
    def __init__(self, names):
        self.layers = _FakeLayers(_FakeLayer(n) for n in names)


def test_clear_patch_viewer_layers_removes_old_patch_layers_only():
    """`_clear_patch_viewer_layers_safely` must remove the previous patch's
    image + work_labels + prediction layers, and leave unrelated layers."""
    viewer = _FakeViewer([
        "patch_3_XY_native_rgb_stack",        # patch image -> remove (patch-name match)
        "patch_3_work_labels_lateral_3d",     # work labels -> remove
        "predictions_overlay",                # prediction prefix -> remove
        "some_unrelated_layer",               # keep
        "another_random_image",               # keep
    ])
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.viewer = viewer
    ctrl.loaded_patch_name = "patch_3"

    ctrl._clear_patch_viewer_layers_safely()

    remaining = {l.name for l in viewer.layers}
    assert remaining == {"some_unrelated_layer", "another_random_image"}, (
        f"unexpected remaining layers: {remaining}"
    )


def test_clear_patch_viewer_layers_safe_when_nothing_matches():
    """No patch layers + empty patch name (first-time Browse) -> no-op, no raise."""
    viewer = _FakeViewer(["unrelated_a", "unrelated_b"])
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.viewer = viewer
    ctrl.loaded_patch_name = ""

    ctrl._clear_patch_viewer_layers_safely()  # must not raise

    assert {l.name for l in viewer.layers} == {"unrelated_a", "unrelated_b"}

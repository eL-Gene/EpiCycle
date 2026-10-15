"""Tests for the Step-3 run-switch safety guard + orchestrator (Phase 3a).

Two surfaces:

1. ``_switch_run_guard.check_switch_safety`` — the PURE verdict function
   (BLOCKED on any of 11 in-flight flags, NEEDS_CONFIRMATION when the 3D
   annotator is open, SAFE otherwise; never raises on a partial controller).

2. ``FusionReconstructionController._switch_active_run`` — the orchestrator
   SEQUENCE (same-run no-op; BLOCKED aborts before any teardown; the
   run-epoch bumps; the per-patch in-memory state resets) bound to a fake
   controller via the unbound method.  No real Qt.

3. ``_warn_patch_run_slug_collisions`` — the namespace guard (a patch stem
   that case-fold collides with an existing run slug emits a WARN; loading
   still proceeds).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from napari_epicycle.fusion_reconstruction import _switch_run_guard as g
from napari_epicycle.fusion_reconstruction._switch_run_guard import (
    SwitchSafety,
    check_switch_safety,
    STEP3_BLOCK_FLAGS,
)


# ---------------------------------------------------------------------------
# 1. Guard verdicts
# ---------------------------------------------------------------------------

def _safe_controller(**overrides):
    """A controller with every block flag False + no annotator open."""
    ns = types.SimpleNamespace()
    for flag in STEP3_BLOCK_FLAGS:
        setattr(ns, flag, False)
    ns._get_open_annotator_widget = lambda: None
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_block_flag_count_is_fourteen():
    """A drift guard, not a magic number: adding a flag here is a deliberate
    widening of what blocks a run switch, so it should have to be typed twice.

    Eleven worker flags, plus the two MODE flags added 2026-09-09 (Brick R1
    audit): `_reshape_running` and `_add_mode_running`.  A run switch runs
    `_guard_unsaved_changes()` BEFORE the annotator close that would restore
    the punched-out cell, so without the first one a user who answers "Save"
    mid-split writes a volume with a HOLE in it to disk.  `_add_mode_running`
    had the same gap since 2026-06-30 (a staged Add is lost; nothing is
    corrupted).
    """
    # 14 since Brick B1 (2026-09-22): `_refine_running` joined them.
    assert len(STEP3_BLOCK_FLAGS) == 14
    assert "_qc_extract_running" in STEP3_BLOCK_FLAGS
    assert "_reshape_running" in STEP3_BLOCK_FLAGS
    assert "_add_mode_running" in STEP3_BLOCK_FLAGS


def test_every_block_flag_has_a_plain_language_reason():
    """A blocked switch shows the reason; a flag with no entry shows nothing."""
    from napari_epicycle.fusion_reconstruction._switch_run_guard import (
        _FLAG_REASONS,
    )

    for flag in STEP3_BLOCK_FLAGS:
        assert _FLAG_REASONS.get(flag), "no reason string for {}".format(flag)


@pytest.mark.parametrize("flag", STEP3_BLOCK_FLAGS)
def test_each_flag_blocks(flag):
    ctrl = _safe_controller()
    setattr(ctrl, flag, True)
    safety, reasons = check_switch_safety(ctrl, viewer=None)
    assert safety == SwitchSafety.BLOCKED
    assert reasons  # non-empty headline


def test_annotator_open_needs_confirmation():
    ctrl = _safe_controller(_get_open_annotator_widget=lambda: object())
    safety, reasons = check_switch_safety(ctrl, viewer=None)
    assert safety == SwitchSafety.NEEDS_CONFIRMATION
    assert reasons


def test_clean_controller_is_safe():
    safety, reasons = check_switch_safety(_safe_controller(), viewer=None)
    assert safety == SwitchSafety.SAFE
    assert reasons == []


def test_block_takes_precedence_over_annotator_open():
    ctrl = _safe_controller(_get_open_annotator_widget=lambda: object())
    ctrl._save_in_flight = True
    safety, _ = check_switch_safety(ctrl, viewer=None)
    assert safety == SwitchSafety.BLOCKED


def test_partial_controller_never_raises():
    # A bare object missing every attr/method must still yield a verdict.
    bare = types.SimpleNamespace()
    safety, reasons = check_switch_safety(bare, viewer=None)
    assert safety == SwitchSafety.SAFE  # nothing in flight, no annotator getter


def test_annotator_getter_that_raises_falls_through_to_safe():
    def _boom():
        raise RuntimeError("kaboom")
    ctrl = _safe_controller(_get_open_annotator_widget=_boom)
    safety, _ = check_switch_safety(ctrl, viewer=None)
    assert safety == SwitchSafety.SAFE


# ---------------------------------------------------------------------------
# 2. Orchestrator sequence (real method bound to a fake controller)
# ---------------------------------------------------------------------------

def _fake_orchestrator_controller(active_run="run_a"):
    """A SimpleNamespace fake with the helpers the orchestrator calls mocked."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    fake = types.SimpleNamespace()
    fake._switch = FusionReconstructionController._switch_active_run

    fake.working_dir = "/ws"
    fake.viewer = types.SimpleNamespace(layers=[])
    fake.ui = MagicMock()

    # state manager
    sm = MagicMock()
    sm.get_active_run_id.return_value = active_run
    sm.loaded_stack_dir = ""
    fake.state_manager = sm

    # block flags + re-entry guard + run epoch + tokens + per-patch state
    for flag in STEP3_BLOCK_FLAGS:
        setattr(fake, flag, False)
    fake._run_switch_in_flight = False
    fake._run_epoch = 0
    fake._compute_dispatch_token = 0
    fake._patch_load_token = 0
    fake._commit_triage_token = 0
    fake._save_curation_token = 0
    fake._qc_features_df = "STALE_DF"
    fake._qc_view_patch_name = "old_patch"
    fake._cached_source_path = "/old/path"
    fake._qc_view_layers = ["lyr"]
    fake._curation_dirty = True
    fake._tiers_dirty = True

    # helper methods (all mocked)
    fake._get_open_annotator_widget = MagicMock(return_value=None)
    fake._close_3d_curation_annotator = MagicMock()
    fake._reset_workspace_scoped_caches = MagicMock()
    fake._guard_unsaved_changes = MagicMock(return_value=True)
    fake._show_switch_blocked_message = MagicMock()
    fake._confirm_switch = MagicMock(return_value=True)
    fake._run_display_name_for = MagicMock(return_value="Display")
    fake._selected_curation_patch = MagicMock(return_value="")
    fake._open_3d_curation_annotator = MagicMock()
    fake._append_log = MagicMock()
    fake._load_stack_dir = MagicMock()
    for fn in (
        "_refresh_patch_state_list", "_clear_model_widgets",
        "_restore_persisted_model_selection", "_discover_workspace_models",
        "_refresh_patch_list_colors", "_refresh_summary_dashboard",
        "_populate_refinement_source_combo", "_refresh_refinement_patch_list",
        "_refresh_per_orientation_summary", "_refresh_mode_hint",
        "_refresh_step3_lock", "_refresh_run_picker",
        "_sync_loaded_patch_marker",      # A3: the loaded-row marker
    ):
        setattr(fake, fn, MagicMock())
    fake._restore_persisted_model_selection.return_value = True
    return fake


def _run_switch(fake, run_id):
    fake._switch(fake, run_id)


def test_same_run_is_noop():
    fake = _fake_orchestrator_controller(active_run="run_b")
    _run_switch(fake, "run_b")
    fake.state_manager.switch_active_run.assert_not_called()
    fake._close_3d_curation_annotator.assert_not_called()
    assert fake._run_epoch == 0


def test_no_workspace_is_noop():
    fake = _fake_orchestrator_controller()
    fake.working_dir = ""
    _run_switch(fake, "run_b")
    fake.state_manager.switch_active_run.assert_not_called()
    assert fake._run_epoch == 0


def test_reentry_guard_blocks():
    fake = _fake_orchestrator_controller()
    fake._run_switch_in_flight = True
    _run_switch(fake, "run_b")
    fake.state_manager.switch_active_run.assert_not_called()


def test_blocked_aborts_before_any_teardown():
    fake = _fake_orchestrator_controller()
    fake._prediction_running = True
    _run_switch(fake, "run_b")
    fake._show_switch_blocked_message.assert_called_once()
    fake._close_3d_curation_annotator.assert_not_called()
    fake.state_manager.switch_active_run.assert_not_called()
    fake._reset_workspace_scoped_caches.assert_not_called()
    assert fake._run_epoch == 0  # epoch must NOT bump on a blocked switch


def test_happy_path_bumps_epoch_and_resets_state():
    fake = _fake_orchestrator_controller(active_run="run_a")
    _run_switch(fake, "run_b")
    # epoch bumped exactly once
    assert fake._run_epoch == 1
    # per-patch in-memory state cleared
    assert fake._qc_features_df is None
    assert fake._qc_view_patch_name == ""
    assert fake._cached_source_path == ""
    assert fake._qc_view_layers == []
    assert fake._curation_dirty is False
    assert fake._tiers_dirty is False
    # patch-scoped tokens bumped (stale-result discard)
    assert fake._compute_dispatch_token == 1
    assert fake._patch_load_token == 1
    assert fake._commit_triage_token == 1
    assert fake._save_curation_token == 1
    # state manager actually re-pointed + caches reset + annotator closed
    fake.state_manager.switch_active_run.assert_called_once_with("run_b")
    fake._reset_workspace_scoped_caches.assert_called_once()
    fake._close_3d_curation_annotator.assert_called_once()
    # re-entry flag cleared in finally
    assert fake._run_switch_in_flight is False


def test_annotator_reopened_only_when_was_open_and_patch_selected():
    fake = _fake_orchestrator_controller()
    fake._get_open_annotator_widget = MagicMock(return_value=object())
    fake._selected_curation_patch = MagicMock(return_value="patch_001")
    _run_switch(fake, "run_b")
    fake._guard_unsaved_changes.assert_called_once()
    fake._confirm_switch.assert_called_once()
    fake._open_3d_curation_annotator.assert_called_once()


def test_needs_confirmation_declined_aborts():
    fake = _fake_orchestrator_controller()
    fake._get_open_annotator_widget = MagicMock(return_value=object())
    fake._confirm_switch = MagicMock(return_value=False)
    _run_switch(fake, "run_b")
    fake.state_manager.switch_active_run.assert_not_called()
    assert fake._run_epoch == 0


def test_unsaved_guard_declined_aborts():
    fake = _fake_orchestrator_controller()
    fake._get_open_annotator_widget = MagicMock(return_value=object())
    fake._guard_unsaved_changes = MagicMock(return_value=False)
    _run_switch(fake, "run_b")
    fake._confirm_switch.assert_not_called()
    fake.state_manager.switch_active_run.assert_not_called()
    assert fake._run_epoch == 0


# ---------------------------------------------------------------------------
# 2b. FIX 1 — unsaved-changes guard runs UNCONDITIONALLY (SAFE path too)
# ---------------------------------------------------------------------------

def test_fix1_safe_switch_runs_unsaved_guard_exactly_once():
    # Annotator CLOSED (default) → SAFE path; dirty tiers set outside the dock.
    fake = _fake_orchestrator_controller()
    fake._tiers_dirty = True
    _run_switch(fake, "run_b")
    # Guard ran exactly once on a SAFE switch (no annotator confirm prompt).
    fake._guard_unsaved_changes.assert_called_once()
    fake._confirm_switch.assert_not_called()
    # Proceeds when the guard returns True.
    fake.state_manager.switch_active_run.assert_called_once_with("run_b")


def test_fix1_safe_switch_guard_declined_aborts_and_preserves_dirty():
    fake = _fake_orchestrator_controller()
    fake._tiers_dirty = True
    fake._curation_dirty = True
    fake._guard_unsaved_changes = MagicMock(return_value=False)
    _run_switch(fake, "run_b")
    # Guard called once; switch aborts BEFORE any teardown.
    fake._guard_unsaved_changes.assert_called_once()
    fake._close_3d_curation_annotator.assert_not_called()
    fake._reset_workspace_scoped_caches.assert_not_called()
    fake.state_manager.switch_active_run.assert_not_called()
    # Dirty flags NOT zeroed (step (d) never ran).
    assert fake._tiers_dirty is True
    assert fake._curation_dirty is True
    assert fake._run_epoch == 0


# ---------------------------------------------------------------------------
# 2c. FIX 2 — hard-clear controller-side curation surfaces on switch
# ---------------------------------------------------------------------------

def test_fix2_hard_clear_curation_surfaces_called_on_happy_path():
    fake = _fake_orchestrator_controller()
    fake._hard_clear_curation_surfaces = MagicMock()
    _run_switch(fake, "run_b")
    fake._hard_clear_curation_surfaces.assert_called_once()


# ---------------------------------------------------------------------------
# 2d. FIX 3 — re-select the new run's resume patch before transparent reopen
# ---------------------------------------------------------------------------

def _bind_resume_resolver(fake):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    fake._resolve_resume_patch_for_run = types.MethodType(
        FusionReconstructionController._resolve_resume_patch_for_run, fake
    )


def test_fix3_reopen_reselects_resume_patch_in_new_run():
    fake = _fake_orchestrator_controller()
    fake._get_open_annotator_widget = MagicMock(return_value=object())  # was open
    _bind_resume_resolver(fake)
    fake.state_manager.get_resume_patch.return_value = ("patch_007", "curating")
    fake._select_curation_bucket_item = MagicMock()
    # After re-select, the bucket reports the resume patch as selected.
    fake._selected_curation_patch = MagicMock(return_value="patch_007")
    _run_switch(fake, "run_b")
    fake._select_curation_bucket_item.assert_called_once_with("patch_007")
    fake._open_3d_curation_annotator.assert_called_once()


def test_fix3_no_resume_patch_stays_closed():
    fake = _fake_orchestrator_controller()
    fake._get_open_annotator_widget = MagicMock(return_value=object())  # was open
    _bind_resume_resolver(fake)
    fake.state_manager.get_resume_patch.return_value = None
    fake.state_manager.active_patch = None
    fake.ui._curation_bucket.count.return_value = 0  # empty bucket fallback
    fake._select_curation_bucket_item = MagicMock()
    fake._selected_curation_patch = MagicMock(return_value="")
    _run_switch(fake, "run_b")
    fake._select_curation_bucket_item.assert_not_called()
    fake._open_3d_curation_annotator.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Namespace guard
# ---------------------------------------------------------------------------

def _fake_namespace_controller(run_ids):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    fake = types.SimpleNamespace()
    fake._warn = FusionReconstructionController._warn_patch_run_slug_collisions
    sm = MagicMock()
    sm.list_run_ids.return_value = run_ids
    fake.state_manager = sm
    fake.logs = []
    fake._append_log = lambda msg: fake.logs.append(msg)
    return fake


def test_patch_stem_collides_with_run_slug_warns():
    fake = _fake_namespace_controller(["default", "cp_membrane"])
    fake._warn(fake, ["cp_membrane.tif", "patch_002.tif"])
    assert any("[WARN]" in m and "cp_membrane" in m for m in fake.logs)


def test_patch_stem_collision_is_case_insensitive():
    fake = _fake_namespace_controller(["default", "CP_Membrane"])
    fake._warn(fake, ["cp_membrane.tif"])
    assert any("[WARN]" in m for m in fake.logs)


def test_no_collision_no_warning():
    fake = _fake_namespace_controller(["default", "cp_membrane"])
    fake._warn(fake, ["patch_001.tif", "patch_002.tif"])
    assert fake.logs == []


def test_default_run_slug_ignored():
    fake = _fake_namespace_controller(["default"])
    fake._warn(fake, ["default.tif"])  # collides only with DEFAULT_RUN_ID
    assert fake.logs == []


def test_namespace_guard_never_raises_on_bad_state_manager():
    fake = types.SimpleNamespace()
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    sm = MagicMock()
    sm.list_run_ids.side_effect = RuntimeError("boom")
    fake.state_manager = sm
    fake._append_log = MagicMock()
    # must not raise
    FusionReconstructionController._warn_patch_run_slug_collisions(
        fake, ["patch.tif"]
    )

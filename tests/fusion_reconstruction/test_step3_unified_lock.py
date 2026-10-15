"""Tests for the unified Step-3 busy lock
(integration/step3-patch-perf, 2026-06-06).

The Step-3 busy lock was unified into a single STATE-DRIVEN helper
``_refresh_step3_lock`` that LOCKS the curate surface + the open 3D annotator +
the workflow step-nav whenever ANY long op is active — a curate busy op
(``_curation_busy``), an async patch load (``_patch_load_in_flight``), a
prediction run (``_prediction_running``), or a reconstruction run
(``_reconstruction_running``).

Goals of this unification:
  (a) the lock also covers the workflow step-nav (``_workflow_shell.lock_step_nav``),
      not just the curation controls; and
  (b) it covers Predict / Reconstruct (lock the curate surface + nav + block
      patch-switch while they run).

Because the lock is recomputed from the live flags on every transition, the
step-nav lock is GUARANTEED to release once all ops finish (guarding against the
``lock_step_nav`` cross-controller leak memory warns about).

Source-introspection + light-fake tests (no Qt display / napari viewer) so they
run in CI alongside the rest of ``tests/fusion_reconstruction/``.
"""

from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# _refresh_step3_lock exists + ORs the four flags + drives nav/annotator/widgets.
# ---------------------------------------------------------------------------

def test_refresh_step3_lock_exists():
    assert hasattr(FusionReconstructionController, "_refresh_step3_lock")


def test_refresh_step3_lock_ors_four_flags_source():
    """The lock must be the OR of the four long-op flags."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_step3_lock
    )
    for flag in (
        "_curation_busy",
        "_patch_load_in_flight",
        "_prediction_running",
        "_reconstruction_running",
    ):
        assert flag in src, (
            f"_refresh_step3_lock must consult {flag} so the lock covers it"
        )


def test_refresh_step3_lock_drives_nav_annotator_widgets_source():
    """The unified lock must touch all three surfaces: curation widgets
    (setEnabled over _CURATION_LOCK_WIDGETS), the open annotator
    (set_load_locked), and the workflow step-nav (lock_step_nav)."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_step3_lock
    )
    assert "_CURATION_LOCK_WIDGETS" in src
    assert "setEnabled" in src
    assert "set_load_locked" in src
    assert "_get_open_annotator_widget" in src
    assert "lock_step_nav" in src
    assert "_workflow_shell" in src
    assert "[LOAD-LOCK]" in src


def test_set_curation_locked_delegates_to_refresh_source():
    """The legacy _set_curation_locked must now just delegate to the
    state-driven _refresh_step3_lock (the actual lock logic lives there)."""
    src = inspect.getsource(
        FusionReconstructionController._set_curation_locked
    )
    assert "_refresh_step3_lock()" in src


def test_set_curation_locked_keeps_locked_param():
    """The `locked` param is retained (advisory/back-compat) so existing
    _busy_show/_busy_hide call sites + their tests keep working."""
    sig = inspect.signature(
        FusionReconstructionController._set_curation_locked
    )
    assert "locked" in sig.parameters


# ---------------------------------------------------------------------------
# _update_button_states refreshes the unified lock (predict/recon → nav lock).
# ---------------------------------------------------------------------------

def test_update_button_states_refreshes_step3_lock_source():
    """Every predict/reconstruct start/finish/cancel goes through
    _update_button_states; it must call _refresh_step3_lock so the curate
    surface + step-nav lock track the pipeline running state."""
    src = inspect.getsource(
        FusionReconstructionController._update_button_states
    )
    assert "_refresh_step3_lock()" in src, (
        "_update_button_states must call _refresh_step3_lock so predict/recon "
        "lock the curate surface + step-nav (and release it when they finish)"
    )
    # The existing cancel-button logic must remain (it stays enabled during a run).
    assert "pipeline_cancel_btn" in src


# ---------------------------------------------------------------------------
# Patch-switch guard blocks during predict/reconstruct.
# ---------------------------------------------------------------------------

def test_switch_guard_includes_predict_reconstruct_source():
    src = inspect.getsource(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "_prediction_running" in src, (
        "_switch_to_curation_patch must block a switch while a prediction runs"
    )
    assert "_reconstruction_running" in src, (
        "_switch_to_curation_patch must block a switch while a reconstruction runs"
    )


def test_switch_guard_returns_early_when_predicting():
    """Live: with _prediction_running True (no patch load / curate busy), the
    switch returns early before touching any surface."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._patch_load_in_flight = False
    c._curation_busy = False
    c._prediction_running = True
    c._reconstruction_running = False
    c._qc_view_patch_name = "other_patch"
    appended = []
    c._append_log = lambda m: appended.append(m)

    # No _hard_clear_curation_surfaces / _guard_unsaved_changes defined on the
    # bare instance — if the guard fails to early-return, this raises
    # AttributeError, surfacing as a test failure.
    c._switch_to_curation_patch("some_patch")

    assert any("[LOAD-LOCK]" in m for m in appended), (
        "the predicting switch should log a [LOAD-LOCK] ignore marker"
    )


def test_switch_guard_returns_early_when_reconstructing():
    """Live: with _reconstruction_running True, the switch returns early."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._patch_load_in_flight = False
    c._curation_busy = False
    c._prediction_running = False
    c._reconstruction_running = True
    c._qc_view_patch_name = "other_patch"
    appended = []
    c._append_log = lambda m: appended.append(m)

    c._switch_to_curation_patch("some_patch")

    assert any("[LOAD-LOCK]" in m for m in appended), (
        "the reconstructing switch should log a [LOAD-LOCK] ignore marker"
    )


# ---------------------------------------------------------------------------
# _CURATION_LOCK_WIDGETS includes predict/reconstruct (mutual exclusion), NOT
# the cancel button.
# ---------------------------------------------------------------------------

def test_curation_lock_widgets_includes_predict_reconstruct():
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "predict_selected_btn" in locked, (
        "a curate busy op must also disable Predict (full mutual exclusion)"
    )
    assert "reconstruct_selected_btn" in locked, (
        "a curate busy op must also disable Reconstruct (full mutual exclusion)"
    )


def test_curation_lock_widgets_excludes_cancel_btn():
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "pipeline_cancel_btn" not in locked, (
        "the cancel button must STAY enabled during predict/recon so the user "
        "can cancel — _update_button_states manages it, not the lock set"
    )


# ---------------------------------------------------------------------------
# Light-fake runtime: _refresh_step3_lock drives shell.lock_step_nav both ways.
# ---------------------------------------------------------------------------

class _W:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, v):
        self.enabled = v


class _UI:
    pass


class _FakeShell:
    def __init__(self):
        self.nav_calls = []

    def lock_step_nav(self, locked):
        self.nav_calls.append(locked)


def _make_lock_controller():
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    ui = _UI()
    # A couple of real lock-set widgets; the rest are getattr-guarded absent.
    ui._curation_bucket = _W()
    ui.predict_selected_btn = _W()
    ui.reconstruct_selected_btn = _W()
    c.ui = ui
    c._get_open_annotator_widget = lambda: None
    shell = _FakeShell()
    c._workflow_shell = shell
    # Default all flags off.
    c._curation_busy = False
    c._patch_load_in_flight = False
    c._prediction_running = False
    c._reconstruction_running = False
    return c, shell


def test_refresh_step3_lock_locks_nav_when_reconstructing():
    c, shell = _make_lock_controller()
    c._reconstruction_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True, (
        "step-nav must lock while a reconstruction runs"
    )
    # Curate widgets + predict/recon buttons disabled.
    assert c.ui._curation_bucket.enabled is False
    assert c.ui.predict_selected_btn.enabled is False
    assert c.ui.reconstruct_selected_btn.enabled is False

    # Clearing the flag releases the nav lock (the leak guard).
    c._reconstruction_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False, (
        "step-nav must RELEASE once the reconstruction flag clears"
    )
    assert c.ui._curation_bucket.enabled is True
    assert c.ui.predict_selected_btn.enabled is True


def test_refresh_step3_lock_locks_nav_when_predicting():
    c, shell = _make_lock_controller()
    c._prediction_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    c._prediction_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False


def test_refresh_step3_lock_locks_nav_when_committing_triage():
    """2026-06-07: an async Commit-Triage persist must lock the curate surface
    + step-nav for its duration (OR'd into the unified lock)."""
    c, shell = _make_lock_controller()
    c._commit_triage_in_flight = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True, (
        "step-nav must lock while a commit-triage persist runs"
    )
    assert c.ui._curation_bucket.enabled is False

    c._commit_triage_in_flight = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False, (
        "step-nav must RELEASE once the commit-triage flag clears"
    )
    assert c.ui._curation_bucket.enabled is True


def test_refresh_step3_lock_ors_save_in_flight_source():
    """2026-06-07: the async Save-Curation flag (_save_in_flight) must be OR'd
    into the unified lock so a Save locks the surface + step-nav."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_step3_lock
    )
    assert "_save_in_flight" in src, (
        "_refresh_step3_lock must consult _save_in_flight so the async Save "
        "locks the curate surface + step-nav for its duration"
    )


def test_refresh_step3_lock_locks_nav_when_saving():
    """2026-06-07: an async Save-Curation persist must lock the curate surface
    + step-nav for its duration (OR'd into the unified lock)."""
    c, shell = _make_lock_controller()
    c._save_in_flight = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True, (
        "step-nav must lock while a Save-curation persist runs"
    )
    assert c.ui._curation_bucket.enabled is False

    c._save_in_flight = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False, (
        "step-nav must RELEASE once the _save_in_flight flag clears"
    )
    assert c.ui._curation_bucket.enabled is True


def test_switch_guard_includes_save_in_flight_source():
    """The patch-switch guard must block a switch while an async Save runs."""
    src = inspect.getsource(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "_save_in_flight" in src, (
        "_switch_to_curation_patch must block a switch while an async Save runs"
    )


def test_refresh_step3_lock_stays_locked_while_any_flag_set():
    """Overlapping ops: clearing ONE flag while another is still set must NOT
    unlock (state-driven OR — no premature unlock)."""
    c, shell = _make_lock_controller()
    c._curation_busy = True
    c._prediction_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    # Curate op finishes but the prediction is still running → stay locked.
    c._curation_busy = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True, (
        "must STAY locked while the prediction flag is still set"
    )

    # Prediction finishes → now release.
    c._prediction_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False


def test_refresh_step3_lock_guards_missing_shell():
    """No _workflow_shell attribute (or None) must not raise."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    ui = _UI()
    c.ui = ui
    c._get_open_annotator_widget = lambda: None
    c._curation_busy = False
    c._patch_load_in_flight = False
    c._prediction_running = False
    c._reconstruction_running = False
    # No _workflow_shell set at all.
    c._refresh_step3_lock()  # must not raise

    # Explicit None shell also fine.
    c._workflow_shell = None
    c._refresh_step3_lock()  # must not raise


# ---------------------------------------------------------------------------
# FIX-1 (audit 2026-06-06) — predict/reconstruct launch is guarded so a
# construction/start exception cannot leak the running flag → nav-lock leak.
# ---------------------------------------------------------------------------

def _launch_block_src(method) -> str:
    """Return the worker-construction + thread.start() region of a launch
    method (everything from the running-flag set onward)."""
    return inspect.getsource(method)


def test_predict_all_wraps_worker_start_in_try_except():
    """If _SerialPredictionWorker construction / wiring / thread.start() raises,
    _prediction_running must be reset to False + _update_button_states called,
    or the step-nav lock leaks forever."""
    src = inspect.getsource(FusionReconstructionController._predict_all)
    assert "try:" in src, "_predict_all must wrap the launch in try/except"
    assert "except Exception" in src
    # The launch block (worker + thread.start) must be inside the try.
    assert "_SerialPredictionWorker(" in src
    assert "thread.start()" in src
    # On failure: running flag cleared + buttons restored.
    assert "self._prediction_running = False" in src, (
        "prediction launch failure must clear _prediction_running (leak guard)"
    )
    assert "_update_button_states()" in src, (
        "prediction launch failure must call _update_button_states"
    )


def test_reconstruct_selected_wraps_worker_start_in_try_except():
    """Same leak guard for the reconstruction launch path."""
    src = inspect.getsource(FusionReconstructionController._reconstruct_selected)
    assert "try:" in src, "_reconstruct_selected must wrap the launch in try/except"
    assert "except Exception" in src
    assert "_SerialReconstructionWorker(" in src
    assert "thread.start()" in src
    assert "self._reconstruction_running = False" in src, (
        "reconstruction launch failure must clear _reconstruction_running"
    )
    assert "_update_button_states()" in src, (
        "reconstruction launch failure must call _update_button_states"
    )


def test_predict_all_failure_clears_flag_before_start_line():
    """Structural: the running-flag reset + _update_button_states sit in the
    except branch AFTER the try (i.e. they are the failure handler, not the
    happy-path button refresh)."""
    src = inspect.getsource(FusionReconstructionController._predict_all)
    # The except branch should appear after thread.start().
    start_idx = src.index("thread.start()")
    except_idx = src.index("except Exception", start_idx - 600)
    assert except_idx > 0
    tail = src[except_idx:]
    assert "self._prediction_running = False" in tail
    assert "_update_button_states()" in tail


def test_reconstruct_selected_failure_clears_flag_in_except():
    src = inspect.getsource(FusionReconstructionController._reconstruct_selected)
    start_idx = src.index("thread.start()")
    except_idx = src.index("except Exception", start_idx - 600)
    assert except_idx > 0
    tail = src[except_idx:]
    assert "self._reconstruction_running = False" in tail
    assert "_update_button_states()" in tail


# ---------------------------------------------------------------------------
# FIX-3 (audit 2026-06-06) — programmatic busy guard at the TOP of both
# predict/reconstruct, mirroring the _switch_to_curation_patch guard.
# ---------------------------------------------------------------------------

def test_predict_all_has_top_busy_guard():
    src = inspect.getsource(FusionReconstructionController._predict_all)
    assert "_curation_busy" in src, (
        "_predict_all must programmatically block entry while a curate op runs"
    )
    assert "_patch_load_in_flight" in src, (
        "_predict_all must programmatically block entry while a patch load runs"
    )
    assert "[LOAD-LOCK]" in src, (
        "_predict_all busy guard must log a [LOAD-LOCK] marker"
    )


def test_reconstruct_selected_has_top_busy_guard():
    src = inspect.getsource(FusionReconstructionController._reconstruct_selected)
    assert "_curation_busy" in src
    assert "_patch_load_in_flight" in src
    assert "[LOAD-LOCK]" in src


def test_predict_busy_guard_returns_early_live():
    """With _curation_busy True, _predict_all returns before touching any UI
    surface (no loaded_stack_dir / ui access)."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._curation_busy = True
    c._patch_load_in_flight = False
    appended = []
    c._append_log = lambda m: appended.append(m)
    # No self.ui / loaded_stack_dir defined — if the guard fails to early-return
    # the next statements raise AttributeError, surfacing as a failure.
    c._predict_all()
    assert any("Busy" in m for m in appended)


def test_reconstruct_busy_guard_returns_early_live():
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._curation_busy = False
    c._patch_load_in_flight = True
    appended = []
    c._append_log = lambda m: appended.append(m)
    c._reconstruct_selected()
    assert any("Busy" in m for m in appended)


# ---------------------------------------------------------------------------
# FIX-4 (audit 2026-06-06) — busy flags initialized in __init__ (hygiene).
# ---------------------------------------------------------------------------

def test_busy_flags_initialized_in_init_source():
    src = inspect.getsource(FusionReconstructionController.__init__)
    assert "self._curation_busy: bool = False" in src, (
        "__init__ must initialize _curation_busy (hygiene — always exists)"
    )
    assert "self._patch_load_in_flight: bool = False" in src, (
        "__init__ must initialize _patch_load_in_flight (hygiene)"
    )


# ===========================================================================
# Model-refinement TRAINING wired into the unified Step-3 busy lock
# (integration/step3-patch-perf, 2026-06-06).
#
#   (a) "Start Training" disabled during curate ops / predict / reconstruct /
#       extraction (extraction handled by the existing direct-setEnabled mutual
#       exclusion; the others via _refinement_running NOT being set + the lock).
#   (b) While training runs, the curate surface + refine config buttons +
#       predict/reconstruct + step-nav + patch-switch are all LOCKED, while the
#       Cancel button stays live.
# ===========================================================================


def test_refinement_running_initialized_in_init_source():
    src = inspect.getsource(FusionReconstructionController.__init__)
    assert "self._refinement_running: bool = False" in src, (
        "__init__ must initialize _refinement_running (hygiene — always exists)"
    )


def test_refresh_step3_lock_ors_refinement_running_source():
    """The unified lock must consult _refinement_running so training locks the
    curate surface + step-nav."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "_refinement_running" in src, (
        "_refresh_step3_lock must OR in _refinement_running so training locks"
    )


def test_refresh_step3_lock_special_cases_cancel_btn_source():
    """Model Refinement look, Brick B2 (I8): Start Training never renames
    itself; refinement_cancel_btn is the ONE live training button, enabled
    only while training runs (special-cased, not in the lock set)."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "refinement_cancel_btn" in src, (
        "_refresh_step3_lock must special-case refinement_cancel_btn"
    )
    assert "_refinement_running" in src
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "refinement_cancel_btn" not in locked


def test_launch_refinement_btn_is_in_lock_set():
    """Start Training greys under ANY Step-3 operation, training included
    (it is no longer the Cancel button): it is in the one lock list."""
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "launch_refinement_btn" in locked


def test_refine_config_buttons_in_lock_set():
    """The config-mutating refine buttons must be in _CURATION_LOCK_WIDGETS so
    they're disabled during ANY Step-3 long op (incl. training)."""
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    for name in (
        "refine_configure_per_orientation_btn",
        "refine_select_all_btn",
        "refine_deselect_all_btn",
        "refine_apply_preset_btn",
        "refine_save_config_btn",
        "refine_load_config_btn",
        "refine_use_for_predict_btn",
        "refine_import_browse_btn",
    ):
        assert name in locked, f"{name} must be in _CURATION_LOCK_WIDGETS"


def test_refine_view_models_btn_not_in_lock_set():
    """refine_view_models_btn is a read-only model browser — leave it usable
    (NOT in the lock set)."""
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "refine_view_models_btn" not in locked


def test_switch_guard_includes_refinement_running_source():
    src = inspect.getsource(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "_refinement_running" in src, (
        "_switch_to_curation_patch must block a switch while training runs"
    )


def test_launch_refinement_has_top_busy_guard_source():
    src = inspect.getsource(FusionReconstructionController._launch_refinement)
    assert "_curation_busy" in src
    assert "_patch_load_in_flight" in src
    assert "_prediction_running" in src
    assert "_reconstruction_running" in src
    assert "[LOAD-LOCK]" in src, (
        "_launch_refinement busy guard must log a [LOAD-LOCK] marker"
    )
    # The guard CONDITION (the `if (...)` block before working_dir) also
    # refuses a second Start while a run is live (fix pass F7): Cancel is its
    # own button now, so this method is never the Cancel path.
    guard_region = src[: src.index("if not self.working_dir")]
    if_idx = guard_region.index("if (")
    cond_block = guard_region[if_idx: guard_region.index("):", if_idx)]
    assert "_refinement_running" in cond_block, (
        "the launch busy-guard CONDITION must refuse a second Start while "
        "_refinement_running"
    )
    for flag in (
        "_curation_busy",
        "_patch_load_in_flight",
        "_prediction_running",
        "_reconstruction_running",
    ):
        assert flag in cond_block, (
            f"the launch busy-guard must block entry on {flag}"
        )


def test_launch_next_refinement_sets_flag_and_refreshes_source():
    """_launch_next_refinement must set _refinement_running True at the commit
    point and refresh the lock; the worker construction + start must be wrapped
    so a launch failure clears the flag (leak guard)."""
    src = inspect.getsource(
        FusionReconstructionController._launch_next_refinement
    )
    assert "self._refinement_running = True" in src, (
        "_launch_next_refinement must set _refinement_running True on commit"
    )
    assert "_refresh_step3_lock()" in src
    assert "_RefinementTrainingWorker(" in src
    assert "thread.start()" in src
    assert "except Exception" in src, (
        "_launch_next_refinement must wrap the launch in try/except (leak guard)"
    )
    # On failure: the launch-fail except (after thread.start()) clears the flag.
    start_idx = src.index("thread.start()")
    tail = src[start_idx:]
    assert "except Exception" in tail, (
        "the launch-failure except must come AFTER thread.start()"
    )
    fail_branch = tail[tail.index("except Exception"):]
    assert "self._refinement_running = False" in fail_branch, (
        "training launch failure must clear _refinement_running (leak guard)"
    )


def test_restore_launch_button_clears_flag_and_refreshes_source():
    """_restore_launch_button is the single training-fully-done funnel — it must
    clear _refinement_running + refresh the lock so the step-nav releases."""
    src = inspect.getsource(
        FusionReconstructionController._restore_launch_button
    )
    assert "self._refinement_running = False" in src, (
        "_restore_launch_button must clear _refinement_running (the funnel)"
    )
    assert "_refresh_step3_lock()" in src


def test_refinement_termination_handlers_funnel_through_restore():
    """All worker termination handlers must funnel through _restore_launch_button
    (the flag-clear funnel) — finished (last orientation), error, cancelled."""
    for name in (
        "_on_refinement_finished",
        "_on_refinement_error",
        "_on_refinement_cancelled",
    ):
        src = inspect.getsource(getattr(FusionReconstructionController, name))
        assert "_restore_launch_button()" in src, (
            f"{name} must call _restore_launch_button (the flag-clear funnel)"
        )


# ---------------------------------------------------------------------------
# Live light-fake: _refresh_step3_lock with _refinement_running drives nav +
# keeps the launch button live; clearing it releases nav.
# ---------------------------------------------------------------------------


def _make_training_lock_controller():
    c, shell = _make_lock_controller()
    # Add the launch button + a refine config button to the fake UI.
    c.ui.launch_refinement_btn = _W()
    c.ui.refinement_cancel_btn = _W()
    c.ui.refine_select_all_btn = _W()
    return c, shell


def test_training_locks_surface_keeps_launch_btn_live():
    c, shell = _make_training_lock_controller()
    c._refinement_running = True
    c._refresh_step3_lock()
    # Step-nav locked.
    assert shell.nav_calls[-1] is True
    # Curate surface + refine config + predict/recon disabled.
    assert c.ui._curation_bucket.enabled is False
    assert c.ui.predict_selected_btn.enabled is False
    assert c.ui.refine_select_all_btn.enabled is False
    # Start Training greys; Cancel is the one live button (I8).
    assert c.ui.launch_refinement_btn.enabled is False, (
        "Start Training must grey while training runs"
    )
    assert c.ui.refinement_cancel_btn.enabled is True, (
        "Cancel must be live while training runs"
    )


def test_training_done_releases_nav_and_relocks_launch_logic():
    c, shell = _make_training_lock_controller()
    c._refinement_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    # Training fully done → flag cleared.
    c._refinement_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False, (
        "step-nav must RELEASE once _refinement_running clears (leak guard)"
    )
    assert c.ui._curation_bucket.enabled is True
    assert c.ui.refine_select_all_btn.enabled is True
    # With nothing locked, the launch button is enabled (Start Training).
    assert c.ui.launch_refinement_btn.enabled is True


def test_launch_btn_disabled_when_other_op_runs_not_training():
    """When some OTHER op holds the lock (predict) but training is NOT running,
    the launch button must be DISABLED (can't start training mid-predict)."""
    c, shell = _make_training_lock_controller()
    c._prediction_running = True
    c._refinement_running = False
    c._refresh_step3_lock()
    assert c.ui.launch_refinement_btn.enabled is False, (
        "Start Training must be disabled while a prediction runs"
    )


def test_training_stays_locked_while_other_flag_set():
    """State-driven OR: training finishing while a predict is still flagged must
    NOT unlock (no premature release)."""
    c, shell = _make_training_lock_controller()
    c._refinement_running = True
    c._prediction_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    # Training finishes but predict still flagged → stay locked.
    c._refinement_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True
    # Launch button disabled now (training not running + still locked by predict).
    assert c.ui.launch_refinement_btn.enabled is False

    c._prediction_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False


def test_switch_guard_returns_early_when_training():
    """Live: with _refinement_running True, the switch returns early."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._patch_load_in_flight = False
    c._curation_busy = False
    c._prediction_running = False
    c._reconstruction_running = False
    c._refinement_running = True
    c._qc_view_patch_name = "other_patch"
    appended = []
    c._append_log = lambda m: appended.append(m)

    c._switch_to_curation_patch("some_patch")

    assert any("[LOAD-LOCK]" in m for m in appended), (
        "the training switch should log a [LOAD-LOCK] ignore marker"
    )


def test_launch_refinement_busy_guard_returns_early_live():
    """With a prediction running, _launch_refinement returns before touching the
    working_dir / staleness / UI surfaces."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._curation_busy = False
    c._patch_load_in_flight = False
    c._prediction_running = True
    c._reconstruction_running = False
    appended = []
    c._append_log = lambda m: appended.append(m)
    # No working_dir / ui defined — if the guard fails to early-return the next
    # statements raise AttributeError, surfacing as a test failure.
    c._launch_refinement()
    assert any("Busy" in m for m in appended)


# ===========================================================================
# FIX-2 (audit 2026-06-06) — training-pair EXTRACTION wired into the unified
# Step-3 busy lock.
#
# Pre-fix extraction self-managed only its own button + manually disabled
# launch_refinement_btn; it did NOT route through _refresh_step3_lock, so:
#   (a) the curate surface / predict / reconstruct / step-nav stayed UNLOCKED
#       during extraction; and
#   (b) the launch_refinement_btn special-case (re-enabled when locked is False)
#       fired mid-extraction whenever a curate op's _busy_hide called
#       _refresh_step3_lock → all flags False → button re-enabled (the H1
#       re-enable hole), letting training start mid-extraction.
#
# Fix: _extraction_running is now a first-class busy flag — ORed into the lock,
# the extract button is special-cased like launch_refinement_btn, the launch
# busy-guard + the patch-switch guard both consult it, and it clears on every
# extraction termination path → _refresh_step3_lock.
# ===========================================================================


def test_extraction_running_initialized_in_init_source():
    src = inspect.getsource(FusionReconstructionController.__init__)
    assert "self._extraction_running: bool = False" in src, (
        "__init__ must initialize _extraction_running (hygiene — always exists)"
    )


def test_refresh_step3_lock_ors_extraction_running_source():
    """The unified lock must consult _extraction_running so extraction locks the
    curate surface + step-nav (closes the re-enable hole)."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "_extraction_running" in src, (
        "_refresh_step3_lock must OR in _extraction_running so extraction locks"
    )


def test_refresh_step3_lock_diagnostic_includes_extract():
    """The [LOAD-LOCK] diagnostic print must report the extract flag."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "extract=" in src, (
        "the [LOAD-LOCK] step3 diagnostic must include extract=..."
    )


def test_refresh_step3_lock_special_cases_extract_btn_source():
    """extract_training_pairs_btn must be special-cased (not in the lock set) so
    it stays ENABLED as 'Cancel Extraction' while ITS OWN extraction runs —
    mirroring launch_refinement_btn during training."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "extract_training_pairs_btn" in src, (
        "_refresh_step3_lock must special-case extract_training_pairs_btn"
    )
    assert (
        "(not locked) or getattr(self, \"_extraction_running\", False)" in src
    ), (
        "extract_training_pairs_btn enable logic must be (not locked) OR "
        "_extraction_running"
    )


def test_extract_btn_not_in_lock_set():
    """extract_training_pairs_btn must NOT be in _CURATION_LOCK_WIDGETS (it is
    special-cased so it can stay live as Cancel Extraction)."""
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "extract_training_pairs_btn" not in locked


def test_switch_guard_includes_extraction_running_source():
    src = inspect.getsource(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "_extraction_running" in src, (
        "_switch_to_curation_patch must block a switch while extraction runs"
    )


def test_launch_refinement_busy_guard_includes_extraction_source():
    """Training must not be able to start mid-extraction — the _launch_refinement
    top busy-guard CONDITION must consult _extraction_running."""
    src = inspect.getsource(FusionReconstructionController._launch_refinement)
    guard_region = src[: src.index("if not self.working_dir")]
    if_idx = guard_region.index("if (")
    cond_block = guard_region[if_idx: guard_region.index("):", if_idx)]
    assert "_extraction_running" in cond_block, (
        "the launch busy-guard CONDITION must block training while extraction runs"
    )


def test_extraction_launch_sets_flag_and_refreshes_source():
    """Both extraction launch paths must set _extraction_running True at the
    button-flip commit point + refresh the lock, and wrap the worker
    construction + start so a launch failure clears the flag (leak guard)."""
    for name in ("_extract_training_pairs", "_extract_training_pairs_for_stems"):
        src = inspect.getsource(getattr(FusionReconstructionController, name))
        assert "self._extraction_running = True" in src, (
            f"{name} must set _extraction_running True on commit"
        )
        assert "_refresh_step3_lock()" in src, (
            f"{name} must refresh the unified lock after setting the flag"
        )
        assert "_ExtractionWorker(" in src
        assert "thread.start()" in src
        assert "except Exception" in src, (
            f"{name} must wrap the launch in try/except (leak guard)"
        )
        # On failure (after thread.start()): the except clears the flag.
        start_idx = src.index("thread.start()")
        tail = src[start_idx:]
        assert "except Exception" in tail
        fail_branch = tail[tail.index("except Exception"):]
        assert "self._extraction_running = False" in fail_branch, (
            f"{name} launch failure must clear _extraction_running (leak guard)"
        )


def test_restore_extract_button_clears_flag_and_refreshes_source():
    """_restore_extract_button is the single extraction-fully-done funnel — it
    must clear _extraction_running + refresh the lock so the step-nav releases."""
    src = inspect.getsource(
        FusionReconstructionController._restore_extract_button
    )
    assert "self._extraction_running = False" in src, (
        "_restore_extract_button must clear _extraction_running (the funnel)"
    )
    assert "_refresh_step3_lock()" in src


def test_cleanup_extraction_thread_clears_flag_source():
    """_cleanup_extraction_thread (thread.finished) is the defensive belt-and-
    suspenders clear — it must also clear _extraction_running + refresh."""
    src = inspect.getsource(
        FusionReconstructionController._cleanup_extraction_thread
    )
    assert "_extraction_running = False" in src, (
        "_cleanup_extraction_thread must defensively clear _extraction_running"
    )
    assert "_refresh_step3_lock()" in src


def test_extraction_termination_handlers_funnel_through_restore():
    """All extraction worker termination handlers must funnel through
    _restore_extract_button (the flag-clear funnel) — finished, error,
    cancelled."""
    for name in (
        "_on_extraction_finished",
        "_on_extraction_error",
        "_on_extraction_cancelled",
    ):
        src = inspect.getsource(getattr(FusionReconstructionController, name))
        assert "_restore_extract_button()" in src, (
            f"{name} must call _restore_extract_button (the flag-clear funnel)"
        )


# ---------------------------------------------------------------------------
# Live light-fake: extraction locks the surface, keeps the extract button live
# as Cancel, but DISABLES Start Training (the H1 mid-extraction regression).
# ---------------------------------------------------------------------------


def _make_extraction_lock_controller():
    c, shell = _make_lock_controller()
    c.ui.launch_refinement_btn = _W()
    c.ui.extract_training_pairs_btn = _W()
    c.ui.refine_select_all_btn = _W()
    return c, shell


def test_extraction_locks_surface_keeps_extract_btn_live_disables_training():
    """H1 regression test: during extraction the curate surface + step-nav lock,
    the extract button stays live as Cancel Extraction, AND Start Training stays
    DISABLED (can't start training mid-extraction via the re-enable hole)."""
    c, shell = _make_extraction_lock_controller()
    c._extraction_running = True
    c._refresh_step3_lock()
    # Step-nav locked.
    assert shell.nav_calls[-1] is True
    # Curate surface + predict/recon + refine config disabled.
    assert c.ui._curation_bucket.enabled is False
    assert c.ui.predict_selected_btn.enabled is False
    assert c.ui.refine_select_all_btn.enabled is False
    # Extract button STAYS enabled (it is "Cancel Extraction" right now).
    assert c.ui.extract_training_pairs_btn.enabled is True, (
        "extract button must stay live as Cancel Extraction during extraction"
    )
    # Start Training STAYS DISABLED — the H1 re-enable hole the audit flagged.
    assert c.ui.launch_refinement_btn.enabled is False, (
        "Start Training must stay DISABLED mid-extraction (no re-enable hole)"
    )


def test_extraction_done_releases_nav_and_buttons():
    c, shell = _make_extraction_lock_controller()
    c._extraction_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    # Extraction fully done → flag cleared.
    c._extraction_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False, (
        "step-nav must RELEASE once _extraction_running clears (leak guard)"
    )
    assert c.ui._curation_bucket.enabled is True
    assert c.ui.refine_select_all_btn.enabled is True
    # Both dual-purpose buttons enabled again.
    assert c.ui.extract_training_pairs_btn.enabled is True
    assert c.ui.launch_refinement_btn.enabled is True


def test_extract_btn_disabled_when_other_op_runs_not_extracting():
    """When some OTHER op holds the lock (predict) but extraction is NOT running,
    the extract button must be DISABLED (can't start extraction mid-predict)."""
    c, shell = _make_extraction_lock_controller()
    c._prediction_running = True
    c._extraction_running = False
    c._refresh_step3_lock()
    assert c.ui.extract_training_pairs_btn.enabled is False, (
        "Extract must be disabled while a prediction runs"
    )


def test_extraction_stays_locked_while_other_flag_set():
    """State-driven OR: extraction finishing while a curate op is still flagged
    must NOT unlock (no premature release)."""
    c, shell = _make_extraction_lock_controller()
    c._extraction_running = True
    c._curation_busy = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    # Extraction finishes but curate op still flagged → stay locked.
    c._extraction_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True
    # Extract button disabled now (not extracting + still locked by curate op).
    assert c.ui.extract_training_pairs_btn.enabled is False

    c._curation_busy = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False


def test_switch_guard_returns_early_when_extracting():
    """Live: with _extraction_running True, the switch returns early."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._patch_load_in_flight = False
    c._curation_busy = False
    c._prediction_running = False
    c._reconstruction_running = False
    c._refinement_running = False
    c._extraction_running = True
    c._qc_view_patch_name = "other_patch"
    appended = []
    c._append_log = lambda m: appended.append(m)

    c._switch_to_curation_patch("some_patch")

    assert any("[LOAD-LOCK]" in m for m in appended), (
        "the extracting switch should log a [LOAD-LOCK] ignore marker"
    )


# ===========================================================================
# FIX-1 (audit 2026-06-06) — the _launch_next_refinement try now covers the
# PRE-COMMIT body (collect / split / config / init-resolve / mkdir), not just
# worker construction.  On the axial pass _refinement_running is already True;
# a pre-commit exception (fired via singleShot) would otherwise strand the flag.
# ===========================================================================


def test_launch_next_refinement_try_precedes_precommit_body():
    """The try: must come BEFORE _collect_training_pairs + bundle_dir.mkdir so a
    pre-commit failure is caught (and the except clears the flag)."""
    src = inspect.getsource(
        FusionReconstructionController._launch_next_refinement
    )
    try_idx = src.index("try:")
    # Pre-commit markers must appear AFTER the try opens.
    assert src.index("_collect_training_pairs(") > try_idx, (
        "_collect_training_pairs must be INSIDE the launch try (FIX-1)"
    )
    assert src.index("split_training_data(") > try_idx, (
        "split_training_data must be INSIDE the launch try (FIX-1)"
    )
    assert src.index("_refinement_launch_snapshot") > try_idx, (
        "the launch-snapshot read (Brick A) must be INSIDE the launch try (FIX-1)"
    )
    assert src.index("snap[\"inits\"]") > try_idx, (
        "the snapshot init read must be INSIDE the launch try (FIX-1)"
    )
    assert src.index(".mkdir(") > try_idx, (
        "bundle_dir.mkdir must be INSIDE the launch try (FIX-1)"
    )
    # And the flag is set + lock refreshed BEFORE the try opens (so the except
    # has a flag to clear).
    flag_idx = src.index("self._refinement_running = True")
    assert flag_idx < try_idx, (
        "the running flag must be set BEFORE the try so the except can clear it"
    )


def test_launch_next_refinement_except_clears_flag_and_drops_queue():
    """The single except must clear the flag, drop the queue, and restore the
    button (which refreshes the lock)."""
    src = inspect.getsource(
        FusionReconstructionController._launch_next_refinement
    )
    except_branch = src[src.index("except Exception"):]
    assert "self._refinement_running = False" in except_branch
    assert "self._refinement_queue = []" in except_branch
    assert "self._restore_launch_button()" in except_branch


def test_launch_next_refinement_failfix5_no_redundant_refresh_in_except():
    """FIX-5: the launch-fail except must NOT CALL _refresh_step3_lock() directly
    — _restore_launch_button() already refreshes (drop the redundant call).

    Comment lines are stripped before the check so the FIX-5 rationale comment
    (which mentions the method name in prose) doesn't trip the substring match —
    we only care about an actual CODE call, not a mention in a comment.
    """
    src = inspect.getsource(
        FusionReconstructionController._launch_next_refinement
    )
    except_branch = src[src.index("except Exception"):]
    code_lines = [
        ln for ln in except_branch.splitlines()
        if not ln.lstrip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert "_refresh_step3_lock()" not in code_only, (
        "FIX-5: drop the redundant _refresh_step3_lock() CALL in the launch-fail "
        "except (_restore_launch_button already refreshes)"
    )


# ===========================================================================
# FIX-3 (audit 2026-06-06) — Cancel during the lateral→axial singleShot gap.
# When the worker is None but _refinement_running is True, Cancel must drop the
# queue + restore the button so the deferred _launch_next_refinement bails.
# ===========================================================================


def test_cancel_refinement_handles_worker_none_gap_source():
    src = inspect.getsource(FusionReconstructionController._cancel_refinement)
    assert "_refinement_running" in src, (
        "_cancel_refinement must handle the worker-None-but-running gap"
    )
    assert "self._refinement_queue = []" in src, (
        "Cancel in the gap must drop the queue so the deferred launch bails"
    )
    assert "_restore_launch_button()" in src, (
        "Cancel in the gap must restore the button (clears flag + refreshes)"
    )


def test_cancel_refinement_gap_live_drops_queue_and_restores():
    """Live: worker is None but _refinement_running True + queue holds the axial
    job → Cancel drops the queue and restores the button (clearing the flag)."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._refinement_worker = None
    c._refinement_running = True
    c._refinement_queue = ["Fine-tune Axial"]
    appended = []
    c._append_log = lambda m: appended.append(m)
    restored = {"called": False}

    def _fake_restore():
        restored["called"] = True
        c._refinement_running = False

    c._restore_launch_button = _fake_restore
    # Minimal fake UI so the status-label setText path is exercised but guarded.
    c.ui = _UI()
    c.ui.refinement_status_label = _W()
    c.ui.refinement_status_label.setText = lambda *_a, **_k: None

    c._cancel_refinement()

    assert c._refinement_queue == [], "Cancel in the gap must clear the queue"
    assert restored["called"], "Cancel in the gap must call _restore_launch_button"


def test_cancel_refinement_with_live_worker_requests_cancel():
    """Live: worker present → request_cancel + disable button (the normal path,
    unchanged by FIX-3)."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)

    class _Worker:
        def __init__(self):
            self.cancelled = False

        def request_cancel(self):
            self.cancelled = True

    w = _Worker()
    c._refinement_worker = w
    c._refinement_running = True
    c.ui = _UI()
    c.ui.refinement_status_label = _W()
    c.ui.refinement_status_label.setText = lambda *_a, **_k: None
    c.ui.launch_refinement_btn = _W()
    c.ui.refinement_cancel_btn = _W()

    c._cancel_refinement()

    assert w.cancelled is True, "live-worker Cancel must call request_cancel"
    assert c.ui.refinement_cancel_btn.enabled is False, (
        "live-worker Cancel must grey Cancel itself (Cancelling...)"
    )
    assert c._refinement_cancel_requested is True
    # Queue must NOT be touched in the live-worker path (the worker's cancelled
    # signal handles teardown).
    assert not hasattr(c, "_refinement_queue") or c._refinement_queue != []


# ===========================================================================
# FIX-4 (audit 2026-06-06) — shutdown() thread-sweep includes the refinement +
# extraction QThreads (quit+wait on app close).
# ===========================================================================


def test_shutdown_sweeps_refinement_and_extraction_threads_source():
    src = inspect.getsource(FusionReconstructionController.shutdown)
    assert "_refinement_thread" in src, (
        "shutdown() must quit+wait the refinement QThread on app close"
    )
    assert "_extraction_thread" in src, (
        "shutdown() must quit+wait the extraction QThread on app close"
    )


def test_shutdown_cancels_training_and_extraction_before_joining():
    """Behaviour: a close mid-training must ASK the workers to stop before it
    quits + waits their threads, and must drop the queued "Both" pass.

    quit() only stops a thread's event loop; a worker inside run() never sees
    it, so without request_cancel() the wait times out with training still
    running.  And a non-empty queue lets the deferred singleShot start the
    axial pass during shutdown.
    """
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    events = []

    class _StateManager:
        def flush(self):
            events.append("flush")

    class _Worker:
        def __init__(self, tag):
            self.tag = tag

        def request_cancel(self):
            events.append(f"{self.tag}.cancel")

    class _Thread:
        def __init__(self, tag):
            self.tag = tag

        def isRunning(self):
            return True

        def quit(self):
            events.append(f"{self.tag}.quit")

        def wait(self, ms):
            events.append(f"{self.tag}.wait({ms})")
            return True

    c.state_manager = _StateManager()
    c._cancel_qc_extraction = lambda quiet=False: None
    c._refinement_worker = _Worker("refine")
    c._refinement_thread = _Thread("refine")
    c._extraction_worker = _Worker("extract")
    c._extraction_thread = _Thread("extract")
    c._refinement_queue = ["Fine-tune Axial"]

    c.shutdown()

    assert c._refinement_queue == [], "the queued Both pass must be dropped"
    for tag in ("refine", "extract"):
        cancel = events.index(f"{tag}.cancel")
        quit_ = events.index(f"{tag}.quit")
        wait = next(i for i, e in enumerate(events) if e.startswith(f"{tag}.wait("))
        assert cancel < quit_ < wait, f"{tag}: want cancel -> quit -> wait, got {events}"
    refine_wait = next(e for e in events if e.startswith("refine.wait("))
    assert int(refine_wait[len("refine.wait("):-1]) > 5000, (
        "training checks its cancel only every few epochs; give it longer "
        f"than the generic 5 s join (got {refine_wait})"
    )


def test_shutdown_disconnects_worker_signals_after_cancel_before_join():
    """After request_cancel() the worker still answers with ``cancelled``
    (or ``finished`` / ``error``), and its handlers write to self.ui --
    which ui.destroyed has already deleted.  shutdown() must cut every
    declared signal of the two workers right after the cancel, before the
    join, so a late emit (or one already queued) never lands.
    """
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    events = []

    class _StateManager:
        def flush(self):
            pass

    class _Signal:
        def __init__(self, label):
            self.label = label

        def disconnect(self, *a):
            events.append(f"{self.label}.disconnect")

    class _Worker:
        def __init__(self, tag, names):
            self.tag = tag
            for n in names:
                setattr(self, n, _Signal(f"{tag}.{n}"))

        def request_cancel(self):
            events.append(f"{self.tag}.cancel")

    class _Thread:
        def __init__(self, tag):
            self.tag = tag
            self.finished = _Signal(f"{tag}.thread.finished")

        def isRunning(self):
            return True

        def quit(self):
            events.append(f"{self.tag}.quit")

        def wait(self, ms):
            events.append(f"{self.tag}.wait")
            return True

    refine_sigs = ("epoch_update", "finished", "error", "log", "cancelled")
    extract_sigs = ("progress", "finished", "error", "log", "cancelled")
    c.state_manager = _StateManager()
    c._cancel_qc_extraction = lambda quiet=False: None
    c._refinement_worker = _Worker("refine", refine_sigs)
    c._refinement_thread = _Thread("refine")
    c._extraction_worker = _Worker("extract", extract_sigs)
    c._extraction_thread = _Thread("extract")
    c._refinement_queue = []

    c.shutdown()

    for tag, names in (("refine", refine_sigs), ("extract", extract_sigs)):
        cancel = events.index(f"{tag}.cancel")
        wait = events.index(f"{tag}.wait")
        for n in names:
            key = f"{tag}.{n}.disconnect"
            assert key in events, f"{key} missing: {events}"
            assert cancel < events.index(key) < wait, (
                f"{key} must come after the cancel and before the join: {events}"
            )
        # The thread's own finished (-> _cleanup_*_thread) is cut after the
        # join, so a cleanup slot cannot land late after ui.destroyed.
        key = f"{tag}.thread.finished.disconnect"
        assert key in events, f"{key} missing: {events}"
        assert events.index(key) > wait, f"{key} must follow the join: {events}"


def test_shutdown_signal_lists_match_the_worker_classes():
    """The names shutdown() cuts are exactly the signals each worker class
    declares, so a signal added later is not left connected."""
    from napari_epicycle.fusion_reconstruction import controller as cmod

    sig_type = type(cmod._RefinementTrainingWorker.finished)
    for cls, names in (
        (cmod._RefinementTrainingWorker, cmod._REFINEMENT_WORKER_SIGNALS),
        (cmod._ExtractionWorker, cmod._EXTRACTION_WORKER_SIGNALS),
    ):
        declared = {
            k for k, v in vars(cls).items() if isinstance(v, sig_type)
        }
        assert declared == set(names), (cls.__name__, declared, names)


def test_shutdown_parks_a_thread_that_outlives_its_join(capsys):
    """A training thread still running after the join must be kept alive
    (and said so), not dropped: Python deleting a running QThread at
    finalization is "QThread: Destroyed while thread is still running",
    the exit-3 close crash."""
    from napari_epicycle.fusion_reconstruction import controller as cmod

    c = FusionReconstructionController.__new__(FusionReconstructionController)

    class _StateManager:
        def flush(self):
            pass

    class _Worker:
        def request_cancel(self):
            pass

    cut = []

    class _FinishedSignal:
        def disconnect(self, *a):
            cut.append("finished")

    class _StuckThread:
        finished = _FinishedSignal()

        def isRunning(self):
            return True

        def quit(self):
            pass

        def wait(self, ms):
            return False  # still running when the join gives up

    worker, thread = _Worker(), _StuckThread()
    c.state_manager = _StateManager()
    c._cancel_qc_extraction = lambda quiet=False: None
    c._refinement_worker = worker
    c._refinement_thread = thread
    c._refinement_queue = []

    try:
        c.shutdown()
        assert (thread, worker) in cmod._LIVE_STEP3_THREADS
        assert cut == ["finished"], "a parked thread's finished must be cut"
        assert (
            "[STEP3] shutdown: _refinement_thread still running after 15 s; parked"
            in capsys.readouterr().out
        )
    finally:
        cmod._LIVE_STEP3_THREADS.discard((thread, worker))


def test_shutdown_survives_a_deleted_thread_wrapper():
    """One thread whose C++ side is already deleted (isRunning() raises
    RuntimeError) must not skip the other joins or the token bumps."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    joined = []

    class _StateManager:
        def flush(self):
            pass

    class _DeletedThread:
        def isRunning(self):
            raise RuntimeError(
                "wrapped C/C++ object of type QThread has been deleted"
            )

    class _Thread:
        def isRunning(self):
            return True

        def quit(self):
            pass

        def wait(self, ms):
            joined.append(ms)
            return True

    c.state_manager = _StateManager()
    c._cancel_qc_extraction = lambda quiet=False: None
    c._refinement_queue = []
    c._prediction_thread = _DeletedThread()     # early in the sweep
    c._save_curation_thread = _Thread()         # last in the sweep
    c._commit_triage_token = 3
    c._patch_load_token = 7

    c.shutdown()  # must not raise

    assert joined == [5000], "the later threads must still be joined"
    assert c._commit_triage_token == 4
    assert c._patch_load_token == 8


def test_shutdown_survives_a_worker_whose_cancel_raises():
    """A request_cancel() that raises (worker already deleted) must not stop
    the rest of shutdown()."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    joined = []

    class _StateManager:
        def flush(self):
            pass

    class _BadWorker:
        def request_cancel(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    class _Thread:
        def isRunning(self):
            return True

        def quit(self):
            pass

        def wait(self, ms):
            joined.append(ms)
            return True

    c.state_manager = _StateManager()
    c._cancel_qc_extraction = lambda quiet=False: None
    c._refinement_worker = _BadWorker()
    c._extraction_worker = _BadWorker()
    c._refinement_thread = _Thread()
    c._extraction_thread = _Thread()
    c._refinement_queue = []

    c.shutdown()

    assert len(joined) == 2, "both threads must still be joined"


# ===========================================================================
# Open-annotator / finalize lock change — audit fixes (2026-06-06).
#
#   FIX-1 — _finalize_curation guard broadened to the FULL op set (peer with
#           _open_3d_curation_annotator).
#   FIX-2 — _undo_last_curation empty-stack guard (the lock force-enables the
#           undo button on unlock; .pop() on an empty stack would IndexError).
#   FIX-3 — _refresh_step3_lock re-runs _refresh_save_button_state on UNLOCK so
#           the save button isn't force-lit on a clean patch after an op.
# ===========================================================================


def test_open_3d_and_finalize_buttons_in_lock_set():
    """Both the open-annotator and finalize buttons must be in
    _CURATION_LOCK_WIDGETS (they gray out during any Step-3 long op)."""
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "open_3d_annotator_button" in locked, (
        "open_3d_annotator_button must be in _CURATION_LOCK_WIDGETS"
    )
    assert "finalize_labels_button" in locked, (
        "finalize_labels_button must be in _CURATION_LOCK_WIDGETS"
    )


def test_open_3d_annotator_busy_guard_ors_six_flags_source():
    """_open_3d_curation_annotator must OR the six op flags and early-return
    (the defense-in-depth programmatic busy guard)."""
    src = inspect.getsource(
        FusionReconstructionController._open_3d_curation_annotator
    )
    for flag in (
        "_curation_busy",
        "_patch_load_in_flight",
        "_prediction_running",
        "_reconstruction_running",
        "_refinement_running",
        "_extraction_running",
    ):
        assert flag in src, (
            f"_open_3d_curation_annotator busy guard must consult {flag}"
        )
    # The guard must early-return with a [LOAD-LOCK] marker.
    assert "[LOAD-LOCK]" in src
    assert "return" in src


def test_finalize_curation_busy_guard_ors_six_flags_source():
    """FIX-1: _finalize_curation must now OR the SAME six op flags as
    _open_3d_curation_annotator (was _patch_load_in_flight-only)."""
    src = inspect.getsource(
        FusionReconstructionController._finalize_curation
    )
    for flag in (
        "_curation_busy",
        "_patch_load_in_flight",
        "_prediction_running",
        "_reconstruction_running",
        "_refinement_running",
        "_extraction_running",
    ):
        assert flag in src, (
            f"_finalize_curation busy guard must consult {flag} (FIX-1)"
        )
    assert "return" in src


def test_finalize_curation_busy_guard_returns_early_live():
    """Live: with _prediction_running True (and no working_dir / ui defined on
    the bare instance), _finalize_curation must early-return before touching any
    surface — if the guard fails the next statement (self.working_dir) raises
    AttributeError, surfacing as a failure."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._curation_busy = False
    c._patch_load_in_flight = False
    c._prediction_running = True
    c._reconstruction_running = False
    c._refinement_running = False
    c._extraction_running = False
    appended = []
    c._append_log = lambda m: appended.append(m)

    c._finalize_curation()

    assert any("loading" in m.lower() or "WARN" in m for m in appended), (
        "blocked finalize should log a warning"
    )


def test_undo_last_curation_has_empty_stack_guard_source():
    """FIX-2: _undo_last_curation must guard against an empty undo stack before
    calling .pop() (the lock force-enables the button on unlock)."""
    src = inspect.getsource(
        FusionReconstructionController._undo_last_curation
    )
    assert "_curation_undo_stack" in src
    # An empty/None check must precede the .pop().
    assert "if not getattr(self, \"_curation_undo_stack\", None)" in src, (
        "_undo_last_curation must early-return when the undo stack is empty"
    )
    guard_idx = src.index(
        "if not getattr(self, \"_curation_undo_stack\", None)"
    )
    # Locate the ACTUAL pop CALL (self._curation_undo_stack.pop()), not the
    # ".pop()" mention inside the rationale comment above the guard.
    pop_idx = src.index("self._curation_undo_stack.pop()")
    assert guard_idx < pop_idx, (
        "the empty-stack guard must come BEFORE the self._curation_undo_stack.pop()"
    )


def test_undo_last_curation_empty_stack_returns_early_live():
    """Live: empty stack → early-return (no .pop() IndexError, no UI access)."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._curation_undo_stack = []
    appended = []
    c._append_log = lambda m: appended.append(m)
    # No _find_active_labels_layer / _refresh_curation_ui defined — if the guard
    # fails to early-return the body raises AttributeError (or IndexError on the
    # pop), surfacing as a test failure.
    c._undo_last_curation()
    assert any("Nothing to undo" in m for m in appended)


def test_refresh_step3_lock_calls_refresh_save_button_state_source():
    """FIX-3: _refresh_step3_lock must re-run _refresh_save_button_state on
    UNLOCK so the save button isn't force-lit on a clean patch."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_step3_lock
    )
    assert "_refresh_save_button_state" in src, (
        "_refresh_step3_lock must re-run the save button's dirty-flag logic "
        "(FIX-3)"
    )
    # It must be gated on NOT locked so it can't re-enable save mid-op.
    assert "if not locked:" in src, (
        "the _refresh_save_button_state call must be gated on `if not locked`"
    )
    call_idx = src.index("_refresh_save_button_state")
    gate_idx = src.rindex("if not locked:", 0, call_idx)
    assert gate_idx >= 0, (
        "the _refresh_save_button_state call must be inside an `if not locked` "
        "block (so it never re-enables save during an op)"
    )


# ===========================================================================
# Batch Refine wired into the unified Step-3 busy lock (2026-06-07).
#
# Batch Refine launches from the 3D annotator (Curation3DAnnotator).  The
# annotator drives a controller flag (_batch_refine_running) via an injected
# callback (_on_annotator_batch_refine_lock) at START and at EVERY terminal
# state (accept / reject / worker error / cancel / dialog close).  The flag is
# OR'd into _refresh_step3_lock so the WHOLE Step-3 surface locks for both the
# worker compute AND the staged-preview window; the navigator-button lock falls
# out of the existing set_load_locked call (no separate nav-lock hack).
# ===========================================================================


def test_batch_refine_running_initialized_in_init_source():
    src = inspect.getsource(FusionReconstructionController.__init__)
    assert "self._batch_refine_running: bool = False" in src, (
        "__init__ must initialize _batch_refine_running (hygiene — always exists)"
    )


def test_refresh_step3_lock_ors_batch_refine_running_source():
    """The unified lock must consult _batch_refine_running so a Batch Refine
    locks the curate surface + step-nav (covers compute + staged preview)."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "_batch_refine_running" in src, (
        "_refresh_step3_lock must OR in _batch_refine_running so Batch Refine "
        "locks the whole Step-3 surface"
    )


def test_refresh_step3_lock_diagnostic_includes_batch():
    """The [LOAD-LOCK] diagnostic print must report the batch flag."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "batch=" in src, (
        "the [LOAD-LOCK] step3 diagnostic must include batch=..."
    )


def test_on_annotator_batch_refine_lock_sets_flag_and_refreshes_source():
    """The injected callback must set _batch_refine_running + recompute the
    unified lock (so the surface tracks the batch op's running/staged state)."""
    src = inspect.getsource(
        FusionReconstructionController._on_annotator_batch_refine_lock
    )
    assert "self._batch_refine_running = bool(active)" in src, (
        "the lock callback must set _batch_refine_running from the `active` arg"
    )
    assert "_refresh_step3_lock()" in src, (
        "the lock callback must recompute the unified lock after setting the flag"
    )


def test_launch_wires_batch_refine_lock_callback_source():
    """The 3D-annotator launch must pass on_batch_refine_lock so the annotator
    can drive the controller flag (mirrors on_batch_refined wiring)."""
    src = inspect.getsource(
        FusionReconstructionController._open_3d_curation_annotator
    )
    assert "on_batch_refine_lock=self._on_annotator_batch_refine_lock" in src, (
        "the launch call must wire on_batch_refine_lock to the controller "
        "callback (mirroring on_batch_refined)"
    )


def test_refresh_step3_lock_locks_nav_when_batch_refining():
    """Live light-fake: with _batch_refine_running True, the curate surface +
    step-nav lock; clearing it releases (the leak guard)."""
    c, shell = _make_lock_controller()
    c._batch_refine_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True, (
        "step-nav must lock while a Batch Refine is running/staged"
    )
    assert c.ui._curation_bucket.enabled is False
    assert c.ui.predict_selected_btn.enabled is False
    assert c.ui.reconstruct_selected_btn.enabled is False

    # Clearing the flag (terminal: accept/reject/error/cancel) releases.
    c._batch_refine_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False, (
        "step-nav must RELEASE once _batch_refine_running clears (leak guard)"
    )
    assert c.ui._curation_bucket.enabled is True
    assert c.ui.predict_selected_btn.enabled is True


def test_batch_refine_stays_locked_while_other_flag_set():
    """State-driven OR: a batch refine resolving while a predict is still
    flagged must NOT unlock (no premature release)."""
    c, shell = _make_lock_controller()
    c._batch_refine_running = True
    c._prediction_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    # Batch refine resolves but predict still running → stay locked.
    c._batch_refine_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True, (
        "must STAY locked while the prediction flag is still set"
    )

    c._prediction_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False


def test_on_annotator_batch_refine_lock_live_sets_flag_and_locks():
    """Live light-fake: invoking the callback with True sets the flag + locks
    the surface; invoking with False clears it + releases."""
    c, shell = _make_lock_controller()
    c._batch_refine_running = False

    c._on_annotator_batch_refine_lock(True)
    assert c._batch_refine_running is True
    assert shell.nav_calls[-1] is True
    assert c.ui._curation_bucket.enabled is False

    c._on_annotator_batch_refine_lock(False)
    assert c._batch_refine_running is False
    assert shell.nav_calls[-1] is False
    assert c.ui._curation_bucket.enabled is True


def test_on_annotator_batch_refine_lock_guards_missing_refresh():
    """A bare instance (no _refresh_step3_lock wired) must not raise — the
    callback try-guards the refresh so a standalone path is safe."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    # No _refresh_step3_lock binding on the bare instance beyond the class
    # method — but ui / workflow_shell are absent, so the method's internal
    # getattr/try guards must absorb everything.  This must not raise.
    c._on_annotator_batch_refine_lock(True)
    assert c._batch_refine_running is True


# ===========================================================================
# "Extract All" QC batch wired into the unified Step-3 busy lock
# (fix/step3-switch-safety-hardening, hole 1, 2026-06-25).
#
# Pre-fix: _extract_features_all launched _QCFeatureBatchWorker WITHOUT setting
# any of the lock flags, so the curate surface / predict / reconstruct / step-nav
# stayed UNLOCKED and a patch switch was NOT blocked while it ran minutes across
# patches — its per-patch set_patch_status would land in the wrong patch context
# (a silent cross-patch race).
#
# Fix: _extract_all_running is a first-class busy flag — ORed into the lock, the
# Extract All button is added to _CURATION_LOCK_WIDGETS (plain no-Cancel op), the
# patch-switch guard consults it, and it clears on every batch termination path
# (all_finished / error / _cleanup_qc_batch_thread, the guaranteed terminal hook)
# -> _refresh_step3_lock.
# ===========================================================================


def test_extract_all_running_initialized_in_init_source():
    src = inspect.getsource(FusionReconstructionController.__init__)
    assert "self._extract_all_running = False" in src, (
        "__init__ must initialize _extract_all_running (hygiene - always exists)"
    )


def test_refresh_step3_lock_ors_extract_all_running_source():
    """The unified lock must consult _extract_all_running so the batch run locks
    the curate surface + step-nav."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "_extract_all_running" in src, (
        "_refresh_step3_lock must OR in _extract_all_running so Extract All locks"
    )


def test_refresh_step3_lock_diagnostic_includes_extract_all():
    """The [LOAD-LOCK] diagnostic print must report the extract_all flag."""
    src = inspect.getsource(FusionReconstructionController._refresh_step3_lock)
    assert "extract_all=" in src, (
        "the [LOAD-LOCK] step3 diagnostic must include extract_all=..."
    )


def test_extract_all_btn_in_lock_set():
    """extract_all_features_btn must BE in _CURATION_LOCK_WIDGETS - it is a plain
    (no-Cancel) batch op, so it is disabled while ANY op locks the surface
    (including its own run via _extract_all_running in the OR)."""
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "extract_all_features_btn" in locked


def test_switch_guard_includes_extract_all_running_source():
    src = inspect.getsource(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "_extract_all_running" in src, (
        "_switch_to_curation_patch must block a switch while Extract All runs"
    )


def test_switch_guard_returns_early_when_extract_all_running():
    """Live: with _extract_all_running True (and the other flags off), the switch
    returns early before touching any surface."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._patch_load_in_flight = False
    c._curation_busy = False
    c._prediction_running = False
    c._reconstruction_running = False
    c._refinement_running = False
    c._extraction_running = False
    c._commit_triage_in_flight = False
    c._save_in_flight = False
    c._batch_refine_running = False
    c._extract_all_running = True
    c._qc_view_patch_name = "other_patch"
    appended = []
    c._append_log = lambda m: appended.append(m)

    c._switch_to_curation_patch("some_patch")

    assert any("[LOAD-LOCK]" in m for m in appended), (
        "the Extract-All switch should log a [LOAD-LOCK] ignore marker"
    )


def test_extract_features_all_sets_flag_and_refreshes_source():
    """_extract_features_all must set _extract_all_running True + refresh the lock
    BEFORE launching the batch worker."""
    src = inspect.getsource(FusionReconstructionController._extract_features_all)
    assert "self._extract_all_running = True" in src, (
        "_extract_features_all must set _extract_all_running True at launch"
    )
    assert "_QCFeatureBatchWorker(" in src
    assert "thread.start()" in src
    # The flag set must precede the worker construction.
    assert src.index("self._extract_all_running = True") < src.index(
        "_QCFeatureBatchWorker("
    ), "the flag must be set BEFORE the worker is constructed"
    assert "_refresh_step3_lock()" in src


def test_on_qc_batch_all_finished_clears_flag_and_refreshes_source():
    src = inspect.getsource(
        FusionReconstructionController._on_qc_batch_all_finished
    )
    assert "self._extract_all_running = False" in src, (
        "_on_qc_batch_all_finished must clear _extract_all_running"
    )
    assert "_refresh_step3_lock()" in src


def test_cleanup_qc_batch_thread_clears_flag_source():
    """_cleanup_qc_batch_thread (thread.finished - the GUARANTEED terminal hook)
    must defensively clear _extract_all_running + refresh."""
    src = inspect.getsource(
        FusionReconstructionController._cleanup_qc_batch_thread
    )
    assert "_extract_all_running = False" in src, (
        "_cleanup_qc_batch_thread must defensively clear _extract_all_running"
    )
    assert "_refresh_step3_lock()" in src


def test_on_qc_extract_error_clears_extract_all_flag_source():
    """The shared error handler must clear _extract_all_running (batch error path)
    before _busy_hide recomputes the lock."""
    src = inspect.getsource(FusionReconstructionController._on_qc_extract_error)
    assert "self._extract_all_running = False" in src, (
        "_on_qc_extract_error must clear _extract_all_running (batch error path)"
    )
    # Cleared BEFORE _busy_hide (whose _refresh_step3_lock recomputes the lock).
    assert src.index("self._extract_all_running = False") < src.index(
        "_busy_hide()"
    ), "the flag must be cleared BEFORE _busy_hide recomputes the lock"


def _make_extract_all_lock_controller():
    c, shell = _make_lock_controller()
    c.ui.extract_all_features_btn = _W()
    # Initialize the full flag set so the OR in _refresh_step3_lock is stable.
    c._refinement_running = False
    c._extraction_running = False
    c._commit_triage_in_flight = False
    c._save_in_flight = False
    c._batch_refine_running = False
    c._extract_all_running = False
    return c, shell


def test_extract_all_locks_surface_and_button():
    """Live light-fake: during Extract All the curate surface + step-nav lock,
    and the Extract All button itself is disabled (plain no-Cancel op)."""
    c, shell = _make_extract_all_lock_controller()
    c._extract_all_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True, (
        "step-nav must lock while Extract All runs"
    )
    assert c.ui._curation_bucket.enabled is False
    assert c.ui.predict_selected_btn.enabled is False
    assert c.ui.reconstruct_selected_btn.enabled is False
    assert c.ui.extract_all_features_btn.enabled is False, (
        "Extract All button must be disabled during its own run (no Cancel mode)"
    )

    # Clearing the flag releases the nav lock + re-enables the surface.
    c._extract_all_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False, (
        "step-nav must RELEASE once _extract_all_running clears (leak guard)"
    )
    assert c.ui._curation_bucket.enabled is True
    assert c.ui.extract_all_features_btn.enabled is True


def test_extract_all_btn_disabled_when_other_op_runs():
    """When some OTHER op holds the lock (predict), the Extract All button must be
    DISABLED (can't launch Extract All mid-predict)."""
    c, shell = _make_extract_all_lock_controller()
    c._prediction_running = True
    c._extract_all_running = False
    c._refresh_step3_lock()
    assert c.ui.extract_all_features_btn.enabled is False, (
        "Extract All must be disabled while a prediction runs"
    )


def test_extract_all_stays_locked_while_other_flag_set():
    """State-driven OR: Extract All finishing while a curate op is still flagged
    must NOT unlock (no premature release)."""
    c, shell = _make_extract_all_lock_controller()
    c._extract_all_running = True
    c._curation_busy = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    # Batch finishes but curate op still flagged -> stay locked.
    c._extract_all_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True, (
        "must STAY locked while the curate-busy flag is still set"
    )

    c._curation_busy = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False


def test_cleanup_qc_batch_thread_live_clears_flag_and_releases():
    """Live: the guaranteed terminal hook clears the flag + releases the lock
    even if the per-event handlers did not."""
    c, shell = _make_extract_all_lock_controller()
    c._extract_all_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True

    c._cleanup_qc_batch_thread()
    assert c._extract_all_running is False
    assert shell.nav_calls[-1] is False, (
        "_cleanup_qc_batch_thread must release the lock (terminal safety net)"
    )


# ---------------------------------------------------------------------------
# Task 2 audit correctives (2026-06-25):
#   - launch-failure guard (HIGH): worker construction wrapped in try/except
#     that clears _extract_all_running so a pre-start exception cannot strand
#     the surface locked forever (mirrors _predict_all FIX-1).
#   - re-entrancy guard (LOW): top-of-method bail if a batch is already in
#     flight (defense-in-depth beyond the disabled button).
#   - button-text restore (LOW cosmetic): _cleanup_qc_batch_thread resets the
#     label so the batch error path does not leave it stuck "Extracting...".
# ---------------------------------------------------------------------------


def test_extract_features_all_wraps_launch_in_try_except():
    """If worker construction / wiring / thread.start() raises, the except must
    clear _extract_all_running + refresh (else the surface strands locked)."""
    src = inspect.getsource(FusionReconstructionController._extract_features_all)
    assert "_QCFeatureBatchWorker(" in src
    assert "thread.start()" in src
    # The launch-failure except (AFTER thread.start) clears the flag + refreshes.
    start_idx = src.index("thread.start()")
    tail = src[start_idx:]
    assert "except Exception" in tail, (
        "the worker launch must be wrapped in try/except (leak guard)"
    )
    fail_branch = tail[tail.index("except Exception"):]
    assert "self._extract_all_running = False" in fail_branch, (
        "Extract All launch failure must clear _extract_all_running (leak guard)"
    )
    assert "_refresh_step3_lock()" in fail_branch, (
        "Extract All launch failure must refresh the lock so the surface unlocks"
    )


def test_extract_features_all_has_reentrancy_guard_source():
    src = inspect.getsource(FusionReconstructionController._extract_features_all)
    assert "if getattr(self, \"_extract_all_running\", False):" in src, (
        "_extract_features_all must bail early if a batch is already running"
    )
    # The guard must precede the working_dir check (it is the very first guard).
    assert src.index("_extract_all_running") < src.index("if not self.working_dir"), (
        "the re-entrancy guard must be the first thing in the method"
    )


def test_extract_features_all_reentrancy_guard_returns_early_live():
    """Live: with _extract_all_running True, the method returns before touching
    working_dir / state_manager / ui (which are absent on the bare instance)."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._extract_all_running = True
    appended = []
    c._append_log = lambda m: appended.append(m)
    c._extract_features_all()
    assert any("[LOAD-LOCK]" in m for m in appended), (
        "a re-entrant Extract All must log a [LOAD-LOCK] ignore marker"
    )


def test_cleanup_qc_batch_thread_resets_button_text_source():
    """The guaranteed terminal hook must restore the button label on every path
    (the error path does not reset it → it would read 'Extracting...' forever)."""
    src = inspect.getsource(
        FusionReconstructionController._cleanup_qc_batch_thread
    )
    assert 'extract_all_features_btn.setText("Extract All")' in src, (
        "_cleanup_qc_batch_thread must restore the Extract All button label"
    )

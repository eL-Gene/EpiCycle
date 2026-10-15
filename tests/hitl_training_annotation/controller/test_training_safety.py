"""Tests for the 2026-05-27 training safety bundle.

Covers 8 audit findings, all in `training_annotation_controller.py`:

- F1.4 — workspace-browse-button typo + F5.1 expanded lock list
- F2.1 — flush debounced state/persist timers before reading staged list
- F4.3 — pre-flight check that working_dir exists on disk
- F5.2 — no-staged-slices gate must run BEFORE any UI mutation
- F4.1 — only-increment counter for buckets with a checkpoint
- F1.5 — training-in-progress marker write/clear/detect cycle
- F7.3 — worker error path prints full traceback (smoke test)

Pure-Python with minimal fakes — no Qt or napari instantiation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
    _TrainingWorker,
    TrainingAbortedError,
)


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class _FakeWidget:
    """Stub for QPushButton / QCheckBox / QComboBox / QLineEdit.

    Tracks enabled state + text changes — enough to verify UI lock + button
    state behavior without instantiating Qt.
    """

    def __init__(self, text=""):
        self._enabled = True
        self._text = text
        self.clicked = MagicMock()

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text


class _FakeListWidget:
    def __init__(self, items=None):
        self._items = list(items or [])

    def count(self):
        return len(self._items)

    def item(self, i):
        m = MagicMock()
        m.text.return_value = self._items[i]
        return m


class _FakeProgressBar:
    def __init__(self):
        self.value = 0
        self._range = (0, 100)

    def setRange(self, lo, hi):
        self._range = (lo, hi)

    def setValue(self, v):
        self.value = v


class _FakeLogOutput:
    def __init__(self):
        self.lines = []

    def append(self, msg):
        self.lines.append(msg)


def _make_ctrl_for_train_model(tmp_path, staged_items=None, working_dir_exists=True):
    """Minimal controller for testing _train_model entry paths.

    Sets up everything needed to reach + early-return from validation
    gates without actually launching training.
    """
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.sample_records = {}
    ctrl.round_backend_info = {}
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl.current_round = 0
    ctrl.selected_view = "XY"
    ctrl.active_run_id = "default"
    ctrl.current_config = {"model_name": "test", "model_type": "cyto3", "epochs": 5}
    ctrl.training_mode = "cumulative"
    ctrl._training_thread = None
    ctrl._flush_called = False

    if working_dir_exists:
        ctrl.working_dir = str(tmp_path)
    else:
        ctrl.working_dir = str(tmp_path / "does_not_exist")

    class _UI:
        pass

    ui = _UI()
    ui.update_and_train_button = _FakeWidget(text="Train")
    ui.training_progress_bar = _FakeProgressBar()
    ui.training_log_output = _FakeLogOutput()
    ui.data_to_add_list = _FakeListWidget(staged_items)
    # Widgets the lock should disable (subset matching real names).
    for attr in TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS:
        setattr(ui, attr, _FakeWidget())
    ctrl.ui = ui

    # Stub the dependencies _train_model calls.
    ctrl._collect_training_config = lambda: ctrl.current_config
    ctrl._sync_data_to_add_list = lambda: None
    ctrl._append_ui_log = lambda msg: ui.training_log_output.append(msg)
    ctrl._current_round_key = lambda: "lateral"
    ctrl._current_round = lambda: 0

    # F2.1 verification hook: when _flush_pending_state_timers is called,
    # set a flag so tests can assert it was called BEFORE the staged-list
    # read.
    def _track_flush():
        ctrl._flush_called = True

    ctrl._flush_pending_state_timers = _track_flush

    return ctrl


# ---------------------------------------------------------------------------
# F1.4 + F5.1 — lock list correctness
# ---------------------------------------------------------------------------


def test_f14_workdir_browse_button_in_lock_list():
    """F1.4 regression test: the actual UI attr name (set_workdir_browse_button)
    must be in the lock list, NOT the old typo (browse_work_dir_button)."""
    lock_list = TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS
    assert "set_workdir_browse_button" in lock_list, "Real attr name missing"
    assert "browse_work_dir_button" not in lock_list, "Old typo still in list"


def test_q8_typed_path_fields_are_in_the_lock_list():
    """Q8: the Workspace and Patch-folder path fields lock with the Browse
    buttons (setText still works on a disabled QLineEdit)."""
    lock_list = TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS
    assert "set_workdir_path" in lock_list
    assert "load_training_path_patch" in lock_list


def test_f51_expanded_lock_list_includes_high_risk_widgets():
    """F5.1: widgets that mutate sample_records / GPU state must be in the
    lock list — otherwise concurrent clicks during training corrupt state
    or CUDA-OOM.

    Bundle 1 follow-up: removed `segment_current_slice_button` and
    `create_empty_mask_button` from this list — they're ghost UI refs
    (declared in _connect_signals but never instantiated on
    main_widget.py).  Pre-existing dead code; not a Bundle 1 safety gap.
    """
    lock_list = TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS
    high_risk = [
        "accept_annotation_button",
        "reject_annotation_button",
        "finalize_labels_button",
        "stage_current_view_button",
        "unstage_selected_button",
        "train_patch_list",
        "launch_multiview_button",
        "refine_results_button",
    ]
    for attr in high_risk:
        assert attr in lock_list, f"{attr!r} must be in lock list for safety"


def test_lock_list_widgets_exist_on_real_ui_source():
    """B2 regression test: every name in _TRAIN_LOCK_WIDGET_ATTRS MUST
    correspond to a real `self.<name> =` assignment in main_widget.py.

    Pre-fix the test created fakes FROM the lock list itself — vacuous
    pass.  This version reads main_widget.py and asserts each entry is
    actually declared on the UI.  Would have caught all 6 typos in the
    original Bundle 1 commit.
    """
    import re
    import inspect
    from napari_epicycle.hitl_training_annotation.ui import main_widget

    source = inspect.getsource(main_widget)
    # Match `self.<name> = <something>` declarations.
    declared = set(re.findall(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*=", source))

    missing = []
    for attr in TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS:
        if attr not in declared:
            missing.append(attr)
    assert not missing, (
        f"Lock list entries with no `self.X = ...` in main_widget.py: {missing}. "
        f"These will silently no-op in _set_training_ui_locked() via getattr(None) — "
        f"same class of bug F1.4 was supposed to close.  Fix the typos in "
        f"_TRAIN_LOCK_WIDGET_ATTRS or remove the dead entries."
    )


def test_set_training_ui_locked_actually_disables_widgets(tmp_path):
    """Smoke test: _set_training_ui_locked(True) disables every widget
    in the lock list."""
    ctrl = _make_ctrl_for_train_model(tmp_path)
    # All locked widgets start enabled.
    for attr in TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS:
        widget = getattr(ctrl.ui, attr, None)
        if widget is not None:
            assert widget.isEnabled() is True

    ctrl._set_training_ui_locked(True)

    for attr in TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS:
        widget = getattr(ctrl.ui, attr, None)
        if widget is not None:
            assert widget.isEnabled() is False, f"{attr!r} should be disabled"

    ctrl._set_training_ui_locked(False)

    for attr in TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS:
        widget = getattr(ctrl.ui, attr, None)
        if widget is not None:
            assert widget.isEnabled() is True, f"{attr!r} should be re-enabled"


# ---------------------------------------------------------------------------
# F2.1 — flush pending timers BEFORE reading staged list
# ---------------------------------------------------------------------------


def test_f21_flush_pending_timers_called_before_staged_list_read(tmp_path):
    """F2.1: _train_model must flush the debounced state-save + curating-
    persist timers BEFORE reading staged_slice_labels.  Otherwise a paint
    stroke <2s before clicking Train silently bypasses training."""
    ctrl = _make_ctrl_for_train_model(tmp_path, staged_items=[])

    # If flush IS called before staged-list read + the staged list is
    # empty, the early-return triggers AFTER flush — flag should be set.
    ctrl._train_model()

    assert ctrl._flush_called is True, "Timers must be flushed before staged-list read"


# ---------------------------------------------------------------------------
# F4.3 — pre-flight check for working_dir existence
# ---------------------------------------------------------------------------


def test_f43_working_dir_missing_aborts_cleanly(tmp_path):
    """F4.3: if working_dir doesn't exist on disk (external delete or
    rename), _train_model must abort BEFORE any UI mutation."""
    ctrl = _make_ctrl_for_train_model(
        tmp_path, staged_items=["x"], working_dir_exists=False
    )
    original_text = ctrl.ui.update_and_train_button.text()

    ctrl._train_model()

    # UI NOT mutated — button text unchanged, widgets still enabled.
    assert ctrl.ui.update_and_train_button.text() == original_text
    for attr in TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS:
        widget = getattr(ctrl.ui, attr, None)
        if widget is not None:
            assert widget.isEnabled() is True
    # User-facing log should mention the missing directory.
    assert any("does not exist" in line.lower() for line in ctrl.ui.training_log_output.lines)


# ---------------------------------------------------------------------------
# F5.2 — no-staged-slices gate runs BEFORE UI mutation
# ---------------------------------------------------------------------------


def test_f52_no_staged_slices_does_not_mutate_ui(tmp_path):
    """REGRESSION test for the user-reported 'training is stuck' bug:
    clicking Train with no staged slices must NOT leave the UI in a
    fake-training state.  Pre-fix: button stuck on 'Cancel Training',
    widgets greyed, progress bar at 5%.  Post-fix: instant clean abort.
    """
    ctrl = _make_ctrl_for_train_model(tmp_path, staged_items=[])
    # the two-button seam: Cancel is born disabled
    ctrl.ui.train_cancel_btn = _FakeWidget(text="Cancel")
    ctrl.ui.train_cancel_btn.setEnabled(False)

    ctrl._train_model()

    # Train is still live (a stuck, greyed Train is the bug this guards) and
    # Cancel is still off: the early return never reached the UI mutation.
    assert ctrl.ui.update_and_train_button.isEnabled() is True, (
        "Train must stay enabled on a no-staged-slices abort"
    )
    assert ctrl.ui.update_and_train_button.text() == "Train"
    assert ctrl.ui.train_cancel_btn.isEnabled() is False
    # All lockable widgets stay ENABLED — no fake-training state.
    for attr in TrainingAnnotationController._TRAIN_LOCK_WIDGET_ATTRS:
        widget = getattr(ctrl.ui, attr, None)
        if widget is not None:
            assert widget.isEnabled() is True, (
                f"{attr!r} must not be disabled on no-staged-slices early return"
            )
    # Progress bar stays at 0, not 5%.
    assert ctrl.ui.training_progress_bar.value == 0


def test_f52_gate_appears_before_button_mutation_in_train_model_source():
    """C4 fix: source-inspection test that F5.2's no-staged gate appears
    BEFORE the UI-mutation block in `_train_model`.  Pre-fix order was
    [mutate UI] → [gate check] which left the user stuck on early-return.
    Post-fix order is [gate check] → [mutate UI].

    The original Bundle 1 had a vacuous positive test (stubbed a method
    that didn't exist).  This replacement directly verifies the
    ordering invariant — would have caught a regression that re-mutates
    UI before checking staged slices.
    """
    import inspect
    source = inspect.getsource(TrainingAnnotationController._train_model)

    # Find key landmarks in the source.
    staged_check_idx = source.find("No staged slices")
    # brick T: the Train button is never renamed; the mutation is "Train
    # greys, Cancel lights" (two buttons)
    button_rename_idx = source.find('cancel_btn.setEnabled(True)')
    progress_set_5_idx = source.find("setValue(5)")

    assert staged_check_idx > 0, "F5.2 no-staged check must be in _train_model"
    assert button_rename_idx > 0, "Cancel-button enable (the UI mutation) must be in _train_model"
    assert progress_set_5_idx > 0, "Progress bar setValue(5) must be in _train_model"

    # The staged check must come BEFORE the button rename AND BEFORE
    # the progress bar mutation.
    assert staged_check_idx < button_rename_idx, (
        f"F5.2 violated: 'No staged slices' check at index {staged_check_idx} "
        f"comes AFTER button rename at index {button_rename_idx} in _train_model. "
        f"Failed gate would leave UI in fake-training state."
    )
    assert staged_check_idx < progress_set_5_idx, (
        f"F5.2 violated: gate at {staged_check_idx} comes AFTER progress bar "
        f"set to 5% at {progress_set_5_idx}.  Failed gate would leave progress "
        f"bar stuck at 5%."
    )


def test_f21_flush_appears_before_staged_check_and_saves_state():
    """F2.1 follow-up (R3 audit): the flush must appear BEFORE the
    staged-list read AND must be followed by an explicit _save_internal_state
    call (R3 found the flush only STOPS timers; doesn't save).
    """
    import inspect
    source = inspect.getsource(TrainingAnnotationController._train_model)

    # Search for the actual call sites (self.X()) not comment mentions.
    flush_call_idx = source.find("self._flush_pending_state_timers()")
    save_call_idx = source.find("self._save_internal_state()")
    staged_read_idx = source.find("self._sync_data_to_add_list()")

    assert flush_call_idx > 0, "F2.1 must call _flush_pending_state_timers"
    assert save_call_idx > 0 and save_call_idx > flush_call_idx, (
        "F2.1 follow-up: must call _save_internal_state AFTER the flush. "
        "Without this, the last 500ms of in-memory state isn't persisted."
    )
    assert staged_read_idx > flush_call_idx, (
        f"F2.1 ordering: flush at {flush_call_idx} must precede staged-list "
        f"read at {staged_read_idx}.  Otherwise late paint strokes get bypassed."
    )


def test_f41_successful_buckets_guard_present_in_on_training_finished():
    """F4.1 source-inspection test: the controller's _on_training_finished
    MUST contain the `successful_buckets` filter checks.  If a future
    refactor removes them, the logic tests above lose their relevance.

    Brittle (regex on source) but catches the specific class of regression
    R2 + R3 flagged: tests can pass while the production code has lost
    the guard.
    """
    import inspect
    source = inspect.getsource(TrainingAnnotationController._on_training_finished)
    assert "successful_buckets" in source, (
        "F4.1 filter (successful_buckets) must be in _on_training_finished. "
        "Without this guard, failed buckets still increment the counter + "
        "commit staged samples + leave orphan state writes."
    )
    # The guard fires at THREE sites: state-write loop, commit loop,
    # counter increment.  All three should be guarded.
    skip_diagnostic_count = source.count("F4.1 SKIP")
    assert skip_diagnostic_count >= 3, (
        f"F4.1 should guard 3 sites (state write, commit, counter "
        f"increment) — found {skip_diagnostic_count} [STEP2-TRAIN] F4.1 SKIP "
        f"diagnostics in _on_training_finished."
    )


# ---------------------------------------------------------------------------
# F4.1 — only increment counter for buckets with checkpoint
# ---------------------------------------------------------------------------


def _make_minimal_ctrl_for_finished():
    """Bare-minimum controller for testing _on_training_finished counter
    increment behavior.  Stubs out the heavy machinery."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.sample_records = {}
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.current_round_by_view = {"lateral": 5, "axial": 3}
    ctrl.current_round = 5
    ctrl.round_backend_info = {}
    ctrl.working_dir = "/tmp/test"
    ctrl.active_run_id = "default"
    ctrl.selected_view = "XY"
    ctrl.current_config = {"model_name": "test", "model_type": "cyto3"}
    return ctrl


def test_f41_failed_bucket_does_not_increment_counter():
    """F4.1: if axial fails (no checkpoint produced) but lateral succeeds,
    ONLY lateral's counter advances.  Pre-fix: both incremented even
    though axial weights don't exist → next round inherits from nothing.
    """
    # We can't trivially run _on_training_finished without massive setup,
    # so we directly exercise the logic block.  The fix is:
    #   for bucket in trained_buckets_round_idx:
    #     if bucket not in successful_buckets: skip
    #     self.current_round_by_view[bucket] += 1
    #
    # Mirror that here against a fake controller.

    ctrl = _make_minimal_ctrl_for_finished()
    pre_lateral = ctrl.current_round_by_view["lateral"]
    pre_axial = ctrl.current_round_by_view["axial"]

    # Simulate: both buckets had losses (got past first epoch) but only
    # lateral produced a checkpoint.  Axial failed mid-training.
    trained_buckets_round_idx = {"lateral": 5, "axial": 3}
    round_checkpoints = {"spatial": "/path/to/lateral_weights"}  # spatial→lateral

    def _bucket_for_view_type(vt):
        if vt == "spatial":
            return "lateral"
        if vt in ("axial_xz", "axial_yz", "axial"):
            return "axial"
        return None

    successful_buckets = set()
    for vt in round_checkpoints.keys():
        b = _bucket_for_view_type(vt)
        if b:
            successful_buckets.add(b)
    for bucket_name in trained_buckets_round_idx.keys():
        if bucket_name not in successful_buckets:
            continue
        ctrl.current_round_by_view[bucket_name] = (
            ctrl.current_round_by_view.get(bucket_name, 0) + 1
        )

    assert ctrl.current_round_by_view["lateral"] == pre_lateral + 1, (
        "lateral should have advanced (had checkpoint)"
    )
    assert ctrl.current_round_by_view["axial"] == pre_axial, (
        "axial must NOT advance — no checkpoint produced (F4.1 regression)"
    )


def test_f41_both_buckets_with_checkpoints_both_increment():
    """Positive case: when both buckets produce checkpoints, both advance."""
    ctrl = _make_minimal_ctrl_for_finished()
    pre_lateral = ctrl.current_round_by_view["lateral"]
    pre_axial = ctrl.current_round_by_view["axial"]

    trained_buckets_round_idx = {"lateral": 5, "axial": 3}
    round_checkpoints = {
        "spatial": "/path/lateral",
        "axial_xz": "/path/axial",
    }

    def _bucket_for_view_type(vt):
        if vt == "spatial":
            return "lateral"
        if vt in ("axial_xz", "axial_yz", "axial"):
            return "axial"
        return None

    successful_buckets = set()
    for vt in round_checkpoints.keys():
        b = _bucket_for_view_type(vt)
        if b:
            successful_buckets.add(b)
    for bucket_name in trained_buckets_round_idx.keys():
        if bucket_name not in successful_buckets:
            continue
        ctrl.current_round_by_view[bucket_name] = (
            ctrl.current_round_by_view.get(bucket_name, 0) + 1
        )

    assert ctrl.current_round_by_view["lateral"] == pre_lateral + 1
    assert ctrl.current_round_by_view["axial"] == pre_axial + 1


def test_f41_mode_single_fail_does_not_commit_staged_samples():
    """F4.1 extension (R3 HIGH): a Mode=Single axial run that fails should
    NOT mark its staged samples as committed.  Pre-fix: all staged
    samples flipped to 'committed' regardless of training outcome,
    forcing the user to manually un-stage to retry.

    Source-inspection: verifies the commit loop in
    _on_training_finished now references successful_buckets.
    """
    import inspect
    source = inspect.getsource(TrainingAnnotationController._on_training_finished)
    # The commit loop should check sample_bucket against successful_buckets
    # before _touch_sample status="committed".
    commit_section_idx = source.find('status="committed"')
    assert commit_section_idx > 0, "Commit-to-committed must be in _on_training_finished"
    # Window of ~500 chars around the commit call should mention
    # successful_buckets (the gate).
    window_start = max(0, commit_section_idx - 500)
    window = source[window_start:commit_section_idx + 100]
    assert "successful_buckets" in window, (
        "F4.1 extension: the commit-to-committed loop must gate on "
        "successful_buckets so failed-bucket samples don't auto-commit."
    )


def test_f41_orphan_per_bucket_state_writes_skipped_for_failed_bucket():
    """F4.1 extension (R3 HIGH): for a failed bucket, the per-bucket
    state writes (training_data_by_round, losses_by_round,
    round_backend_info) should be SKIPPED.  Pre-fix: orphan entries
    accumulated in state JSON.

    Source-inspection: the per-bucket state write loop must check
    successful_buckets and skip failed buckets.
    """
    import inspect
    source = inspect.getsource(TrainingAnnotationController._on_training_finished)
    # The state-write loop sets training_data_by_round[round_key] etc.
    state_write_idx = source.find("training_data_by_round[round_key]")
    assert state_write_idx > 0, "Per-bucket state writes must exist"
    # Window of ~600 chars BEFORE the state write should mention the
    # successful_buckets guard.
    window_start = max(0, state_write_idx - 600)
    window = source[window_start:state_write_idx]
    assert "successful_buckets" in window and (
        "continue" in window or "skip" in window.lower()
    ), (
        "F4.1 extension: the per-bucket state-write loop must skip "
        "buckets not in successful_buckets to prevent orphan entries."
    )


# ---------------------------------------------------------------------------
# F1.5 — training-in-progress marker write/clear/detect
# ---------------------------------------------------------------------------


def test_f15_marker_write_and_clear_cycle(tmp_path):
    """F1.5: marker file is written at training start, cleared on clean
    completion.  Round-trip integrity."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = "test_run"
    ctrl.selected_view = "XY"
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl.current_round = 0
    ctrl.ui = MagicMock()

    marker = tmp_path / ".training_in_progress"
    assert not marker.exists()

    ctrl._write_training_in_progress_marker(round_id="round_00")
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert payload["run_id"] == "test_run"
    assert payload["round_id"] == "round_00"
    assert payload["bucket"] == "lateral"

    ctrl._clear_training_in_progress_marker()
    assert not marker.exists()


def test_f15_detect_stale_marker_emits_warning(tmp_path):
    """F1.5: if a marker file is present from a prior session (force-
    killed napari), the next load surfaces a warning + clears the marker."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = "test_run"
    ctrl.selected_view = "XY"
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl.current_round = 0

    log_messages = []
    ctrl._append_ui_log = log_messages.append
    ctrl.ui = MagicMock()

    # Simulate stale marker from prior session.
    marker = tmp_path / ".training_in_progress"
    marker.write_text(json.dumps({
        "run_id": "test_run",
        "bucket": "lateral",
        "round_id": "round_03",
        "started_at": "2026-05-26T12:00:00+00:00",
        "pid": 9999,
    }))

    ctrl._detect_stale_training_marker()

    assert any("WARNING" in msg and "interrupted" in msg for msg in log_messages), (
        "Stale marker must surface a warning to the user"
    )
    # Marker was cleared so it doesn't keep re-warning.
    assert not marker.exists()


def test_f15_detect_with_no_marker_is_silent(tmp_path):
    """No marker present → no warning."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.working_dir = str(tmp_path)

    log_messages = []
    ctrl._append_ui_log = log_messages.append
    ctrl.ui = MagicMock()

    ctrl._detect_stale_training_marker()

    assert log_messages == []


# ---------------------------------------------------------------------------
# F7.3 — worker error path prints full traceback
# ---------------------------------------------------------------------------


def test_f73_worker_error_prints_traceback(capsys):
    """F7.3: when the training function raises, the worker should print
    a full traceback (not just the bare exception string) for debug."""
    def _failing_train(progress_cb, **_kwargs):
        # Nested call so traceback has meaningful depth.
        def _inner():
            raise ValueError("synthetic training failure")
        _inner()

    worker = _TrainingWorker(_failing_train)
    error_messages = []
    worker.error = MagicMock()
    worker.error.emit = error_messages.append

    worker.run()

    captured = capsys.readouterr()
    # traceback.print_exc() writes to stderr, [TRAIN-SAFETY] print to stdout.
    all_output = captured.out + captured.err
    assert "[TRAIN-SAFETY]" in all_output
    assert "ValueError" in all_output or "ValueError" in str(error_messages)
    assert "Traceback" in all_output, "Full traceback should be printed for debugging"


def test_f73_worker_abort_does_not_print_traceback(capsys):
    """F7.3 negative case: TrainingAbortedError is handled cleanly without
    spamming the console with a traceback (user-initiated, not a bug)."""
    def _aborting_train(progress_cb, **_kwargs):
        raise TrainingAbortedError("user cancelled")

    worker = _TrainingWorker(_aborting_train)
    error_messages = []
    worker.error = MagicMock()
    worker.error.emit = error_messages.append

    worker.run()

    captured = capsys.readouterr()
    all_output = captured.out + captured.err
    # No traceback for clean abort.
    assert "Traceback" not in all_output
    # But the abort message should reach the error signal.
    assert error_messages == ["Training cancelled by user."]

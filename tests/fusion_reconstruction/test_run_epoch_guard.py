"""Phase 3c — RUN-EPOCH cross-run stale-finish guard.

Every async worker in the Step-3 controller captures ``self._run_epoch`` at
LAUNCH; its finish handler discards the result if ``_run_epoch`` changed before
the handler runs (a run-switch bumped it in ``_switch_active_run``).  Without
this, a worker launched against run A could write its result into run B's
in-memory df / on-disk CSV / patch state (the cross-run contamination class).

These tests build bare controllers via ``__new__`` (no Qt / napari / real
state-manager) and call each finish handler directly with (a) a MATCHING epoch
— the result must proceed — and (b) a BUMPED epoch — the result must be
discarded with NO shared-state write, and the op's in-flight flag must not
strand.  The matching-epoch path is the default everywhere (epoch starts at 0
and only bumps on switch), so these guards are byte-identical when no switch
happens.
"""

from __future__ import annotations

import inspect

import pytest

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# Shared Qt-free fakes
# ---------------------------------------------------------------------------


class _RecordingStateManager:
    """Records set_patch_status calls so we can assert they did/didn't fire."""

    def __init__(self):
        self.set_patch_status_calls = []

    def set_patch_status(self, *args, **kwargs):
        self.set_patch_status_calls.append((args, kwargs))

    def get_patch_status(self, patch_name):
        return "reconstructed"


class _FakeBtn:
    def __init__(self):
        self.enabled = None
        self.text = None

    def setEnabled(self, v):
        self.enabled = v

    def setText(self, v):
        self.text = v


class _FakeUI:
    def __init__(self):
        self.extract_features_btn = _FakeBtn()


def _bare_controller():
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._run_epoch = 0
    c._log_messages = []
    c._append_log = lambda m: c._log_messages.append(m)
    c._busy_hidden = []
    c._busy_hide = lambda: c._busy_hidden.append(True)
    return c


# ---------------------------------------------------------------------------
# 1. _on_qc_extract_finished  (instance-attr epoch, on top of patch guard)
# ---------------------------------------------------------------------------


def test_qc_extract_finished_discards_on_run_switch(capsys):
    c = _bare_controller()
    c._qc_view_patch_name = "patch_A"
    c._qc_extract_launch_epoch = 0
    c._qc_features_df = {"sentinel": "RUN_B_DF"}
    c.ui = _FakeUI()
    # A run-switch bumped the epoch after launch.
    c._run_epoch = 1

    FusionReconstructionController._on_qc_extract_finished(
        c, {"sentinel": "RUN_A_DF"}, "patch_A"
    )

    # df must NOT be overwritten with the stale run's result.
    assert c._qc_features_df == {"sentinel": "RUN_B_DF"}
    assert c._busy_hidden, "busy bar must be hidden on the discard path"
    assert c.ui.extract_features_btn.text == "Extract"
    out = capsys.readouterr().out
    assert "[RUN-EPOCH] QC extract STALE FINISH discarded" in out


def test_qc_extract_finished_matching_epoch_passes_guard(capsys):
    c = _bare_controller()
    c._qc_view_patch_name = "patch_A"
    c._qc_extract_launch_epoch = 0
    c._qc_features_df = None
    c.ui = _FakeUI()
    # No switch — epoch matches launch.  The guard must NOT fire (the normal
    # post-guard path will raise on our minimal fake; we only assert the
    # RUN-EPOCH branch did not short-circuit).
    try:
        FusionReconstructionController._on_qc_extract_finished(
            c, {"x": 1}, "patch_A"
        )
    except Exception:
        pass
    out = capsys.readouterr().out
    assert "[RUN-EPOCH] QC extract STALE FINISH discarded" not in out


# ---------------------------------------------------------------------------
# 2. _on_qc_batch_patch_done / _on_qc_batch_all_finished
# ---------------------------------------------------------------------------


def test_qc_batch_patch_done_discards_on_run_switch():
    c = _bare_controller()
    c._qc_batch_launch_epoch = 0
    c._run_epoch = 1
    c.state_manager = _RecordingStateManager()

    FusionReconstructionController._on_qc_batch_patch_done(c, "patch_A", 42)

    assert c.state_manager.set_patch_status_calls == [], (
        "stale-run batch patch-done must NOT write set_patch_status"
    )


def test_qc_batch_patch_done_matching_epoch_writes():
    c = _bare_controller()
    c._qc_batch_launch_epoch = 0
    c._run_epoch = 0
    c.state_manager = _RecordingStateManager()

    FusionReconstructionController._on_qc_batch_patch_done(c, "patch_A", 42)

    assert len(c.state_manager.set_patch_status_calls) == 1, (
        "matching-epoch batch patch-done must write set_patch_status"
    )


# ---------------------------------------------------------------------------
# 3. _launch_async_compute  (local-closure epoch, extends _is_stale)
# ---------------------------------------------------------------------------


class _FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for s in list(self.slots):
            s(*args)


class _FakeComputeWorker:
    def __init__(self, fn):
        self._fn = fn
        self.done = _FakeSignal()
        self.error = _FakeSignal()

    def moveToThread(self, _t):
        pass

    def run(self):
        pass


class _FakeThread:
    def __init__(self):
        self.started = _FakeSignal()
        self.finished = _FakeSignal()

    def start(self):
        pass

    def quit(self):
        pass


def _make_compute_controller(monkeypatch, workers):
    from napari_epicycle.fusion_reconstruction import controller as ctrl_mod

    def _factory(fn):
        w = _FakeComputeWorker(fn)
        workers.append(w)
        return w

    monkeypatch.setattr(ctrl_mod, "_AsyncComputeWorker", _factory)
    monkeypatch.setattr(ctrl_mod, "QThread", _FakeThread)

    c = _bare_controller()
    c._compute_dispatch_token = 0
    c._compute_dispatch_patch = None
    c._qc_view_patch_name = "patchA"
    return c


def test_async_compute_discards_on_run_switch(monkeypatch, capsys):
    workers = []
    c = _make_compute_controller(monkeypatch, workers)
    applied = []

    c._launch_async_compute("TEST", lambda: "R", lambda r: applied.append(r))
    # A run-switch bumps the epoch after launch (token + patch unchanged).
    c._run_epoch = 1
    workers[0].done.slots[0]("R")

    assert applied == [], "result from the prior run must be discarded"
    assert c._busy_hidden, "busy_hide must still run on a run-switch discard"
    assert "run-epoch 0!=1" in capsys.readouterr().out


def test_async_compute_matching_epoch_applies(monkeypatch):
    workers = []
    c = _make_compute_controller(monkeypatch, workers)
    applied = []

    c._launch_async_compute("TEST", lambda: "R", lambda r: applied.append(r))
    # No switch.
    workers[0].done.slots[0]("R")
    assert applied == ["R"]


# ---------------------------------------------------------------------------
# 4. _on_patch_loaded  (lambda-passed run_epoch)
# ---------------------------------------------------------------------------


def test_patch_loaded_discards_on_run_switch(capsys):
    c = _bare_controller()
    c._patch_load_token = 5
    c._patch_load_in_flight = True
    c._run_epoch = 1  # switched after launch (which captured epoch 0)

    FusionReconstructionController._on_patch_loaded(
        c, {"some": "result"}, "patch_A", "patch_A", "stack_patch_A", 5,
        run_epoch=0,
    )

    # The in-flight flag is cleared only PAST the guard → still True ⇒ discarded.
    assert c._patch_load_in_flight is True
    assert c._busy_hidden, "busy bar must be hidden on the run-switch discard"
    assert "[RUN-EPOCH] patch-load discard" in capsys.readouterr().out


def test_patch_loaded_none_epoch_skips_guard(capsys):
    """run_epoch=None (legacy/direct callers) must not trigger the guard."""
    c = _bare_controller()
    c._patch_load_token = 5
    c._patch_load_in_flight = True
    c._run_epoch = 99
    try:
        FusionReconstructionController._on_patch_loaded(
            c, {"some": "result"}, "patch_A", "patch_A", "stack_patch_A", 5,
            run_epoch=None,
        )
    except Exception:
        pass
    assert "[RUN-EPOCH] patch-load discard" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 5. _on_serial_recon_patch_done / _on_serial_recon_complete
# ---------------------------------------------------------------------------


def test_serial_recon_patch_done_discards_on_run_switch():
    c = _bare_controller()
    c._serial_recon_launch_epoch = 0
    c._run_epoch = 1
    c._reconstructing_patches = {"patch_A"}
    c._qc_features_df = "SENTINEL"
    c.state_manager = _RecordingStateManager()

    FusionReconstructionController._on_serial_recon_patch_done(
        c, "patch_A", {"run_id": "r1", "result": None}
    )

    assert c.state_manager.set_patch_status_calls == [], (
        "stale-run recon patch-done must NOT write set_patch_status"
    )
    # _cleanup_stale_qc (which nulls _qc_features_df + deletes the CSV) must
    # NOT have run.
    assert c._qc_features_df == "SENTINEL"


def test_serial_recon_complete_discards_on_run_switch(capsys):
    c = _bare_controller()
    c._serial_recon_launch_epoch = 0
    c._run_epoch = 1
    c._reconstructing_patches = {"patch_A"}
    # If the guard fails to fire, the body touches self.ui / state — absent
    # here — and would raise.  A clean return proves the discard.
    FusionReconstructionController._on_serial_recon_complete(c)
    assert "[RUN-EPOCH] reconstruction all-complete discarded" in (
        capsys.readouterr().out
    )


# ---------------------------------------------------------------------------
# 6. _on_serial_pred_patch_done / _on_serial_pred_complete
# ---------------------------------------------------------------------------


def test_serial_pred_patch_done_discards_on_run_switch():
    c = _bare_controller()
    c._serial_pred_launch_epoch = 0
    c._run_epoch = 1
    c._predicting_patches = {"patch_A"}
    c.state_manager = _RecordingStateManager()

    FusionReconstructionController._on_serial_pred_patch_done(
        c, "patch_A", {"views": {}}
    )

    assert c.state_manager.set_patch_status_calls == [], (
        "stale-run pred patch-done must NOT write set_patch_status"
    )


def test_serial_pred_complete_discards_on_run_switch(capsys):
    c = _bare_controller()
    c._serial_pred_launch_epoch = 0
    c._run_epoch = 1
    c._predicting_patches = {"patch_A"}
    c._pipeline_cancelled = (1, 2)
    c.ui = _FakeUI()
    c.ui.pipeline_cancel_btn = _FakeBtn()
    FusionReconstructionController._on_serial_pred_complete(c)
    assert "[RUN-EPOCH] prediction all-complete discarded" in (
        capsys.readouterr().out
    )
    assert c._pipeline_cancelled is None
    assert c.ui.pipeline_cancel_btn.enabled is False


# ---------------------------------------------------------------------------
# 7. _on_commit_triage_finished  (dispatch_epoch kw)
# ---------------------------------------------------------------------------


def test_commit_triage_finished_discards_on_run_switch(capsys):
    c = _bare_controller()
    c._run_epoch = 1
    c._commit_triage_in_flight = True
    c._tiers_dirty = True
    c._curation_dirty = True
    c.state_manager = _RecordingStateManager()

    FusionReconstructionController._on_commit_triage_finished(
        c, {"ok": True, "triage_map": {}}, 1, "patch_A.tif", {},
        dispatch_epoch=0,
    )

    # Dirty flags must NOT be cleared; status must NOT be written.
    assert c._tiers_dirty is True
    assert c._curation_dirty is True
    assert c.state_manager.set_patch_status_calls == []
    # finally-block cleanup still runs.
    assert c._commit_triage_in_flight is False
    assert c._busy_hidden
    assert "[RUN-EPOCH] commit-triage discard" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 8. _on_save_curation_finished  (dispatch_epoch kw)
# ---------------------------------------------------------------------------


def test_save_curation_finished_discards_on_run_switch(capsys):
    c = _bare_controller()
    c._run_epoch = 1
    c._save_in_flight = True
    c._save_dispatch_patch = "patch_A.tif"
    c._curation_dirty = True
    c._edited_patches = set()

    FusionReconstructionController._on_save_curation_finished(
        c, {"ok": True}, 1, "patch_A.tif", dispatch_epoch=0,
    )

    assert c._curation_dirty is True, "stale-run save must NOT clear dirty"
    assert c._edited_patches == set(), "stale-run save must NOT mark edited"
    # finally cleanup.
    assert c._save_in_flight is False
    assert c._busy_hidden
    assert "[RUN-EPOCH] save-curation discard" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 9. _on_extraction_finished  (training-pair, instance-attr epoch)
# ---------------------------------------------------------------------------


def test_extraction_finished_discards_on_run_switch(capsys):
    c = _bare_controller()
    c._extraction_launch_epoch = 0
    c._run_epoch = 1
    c._extraction_running = True
    restored = []
    c._restore_extract_button = lambda: (
        restored.append(True), setattr(c, "_extraction_running", False)
    )

    FusionReconstructionController._on_extraction_finished(c, {"spatial": 1})

    assert restored, "discard must still funnel through _restore_extract_button"
    assert c._extraction_running is False, "in-flight flag must not strand"
    assert "[RUN-EPOCH] training-pair extraction discarded" in (
        capsys.readouterr().out
    )


# ---------------------------------------------------------------------------
# 10. _on_refinement_finished  (instance-attr epoch)
# ---------------------------------------------------------------------------


def test_refinement_finished_discards_on_run_switch(capsys):
    c = _bare_controller()
    c._refinement_launch_epoch = 0
    c._run_epoch = 1
    c._refinement_running = True
    c._refinement_queue = ["axial"]
    c._last_refined_model_dir = None
    restored = []
    c._restore_launch_button = lambda: (
        restored.append(True), setattr(c, "_refinement_running", False)
    )

    FusionReconstructionController._on_refinement_finished(
        c, {"model_path": "/some/model"}
    )

    assert restored, "discard must funnel through _restore_launch_button"
    assert c._refinement_queue == [], "queued passes must be dropped on discard"
    assert c._last_refined_model_dir is None, "model path must NOT be recorded"
    assert "[RUN-EPOCH] refinement training discarded" in (
        capsys.readouterr().out
    )


# ---------------------------------------------------------------------------
# 11. _switch_in_place_after_extract  (deferred _fire) — source guard
# ---------------------------------------------------------------------------


def test_switch_in_place_captures_and_checks_run_epoch_source():
    src = inspect.getsource(
        FusionReconstructionController._switch_in_place_after_extract
    )
    assert "launch_epoch = getattr(self, \"_run_epoch\", 0)" in src, (
        "_switch_in_place_after_extract must capture the run-epoch before "
        "deferring _fire"
    )
    assert "launch_epoch != getattr(self, \"_run_epoch\", 0)" in src, (
        "_fire must bail if the run switched before it ran"
    )


# ---------------------------------------------------------------------------
# Source-introspection: launch-site captures (the instance-attr workers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name, attr",
    [
        ("_extract_features_current", "_qc_extract_launch_epoch"),
        ("_on_qc_batch_all_finished", "_qc_batch_launch_epoch"),
        ("_on_serial_recon_patch_done", "_serial_recon_launch_epoch"),
        ("_on_serial_pred_patch_done", "_serial_pred_launch_epoch"),
        ("_on_extraction_finished", "_extraction_launch_epoch"),
        ("_on_refinement_finished", "_refinement_launch_epoch"),
    ],
)
def test_handler_references_launch_epoch_attr(method_name, attr):
    src = inspect.getsource(getattr(FusionReconstructionController, method_name))
    assert attr in src, f"{method_name} must reference {attr}"

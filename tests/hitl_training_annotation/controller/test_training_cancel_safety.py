"""Tests for Bundle 2: training cancel safety (2026-05-27).

Covers:
- TrainingAbortedError now derives from BaseException (not Exception).
- TrainingAbortedError is NOT caught by `except Exception` (so it
  propagates through third-party logging chains and Cellpose internals).
- _archive_prior_training_state moves prior weights to a sibling
  .pre_train_archive/ dir.
- _delete_archive removes the archive on training success.
- _restore_prior_training_state moves the archive back on failure/cancel.
- cancel_training updates the button label + disables it.
- cellpose_trainer's narrowed except does not catch TrainingAbortedError.

Pure-Python tests with minimal fakes — no Qt or napari instantiation.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from packages.epicycle_training.exceptions import TrainingAbortedError
from packages.epicycle_training.cellpose_trainer import _EpochLogHandler
from packages.epicycle_training.training_backend_adapter import (
    _archive_prior_training_state,
    _delete_archive,
    _restore_prior_training_state,
    scan_orphan_archives,
    restore_orphan_archive,
    build_epicycle_training_backend,
)


# ---------------------------------------------------------------------------
# TrainingAbortedError base class behavior
# ---------------------------------------------------------------------------


def test_training_aborted_error_inherits_from_base_exception():
    """The architectural choice: BaseException (not Exception)."""
    assert issubclass(TrainingAbortedError, BaseException)
    assert not issubclass(TrainingAbortedError, Exception), (
        "TrainingAbortedError must NOT derive from Exception — that's the "
        "whole point.  An Exception-derived abort gets swallowed by "
        "third-party bare-except blocks (Cellpose's logging chain etc.)."
    )


def test_training_aborted_error_not_caught_by_except_exception():
    """Concrete demo: `except Exception` does NOT catch our abort."""
    caught_by_exception = False
    caught_by_base_exception = False
    try:
        raise TrainingAbortedError("cancelled")
    except Exception:
        caught_by_exception = True
    except BaseException:
        caught_by_base_exception = True

    assert not caught_by_exception, (
        "If Exception catches it, the whole Bundle 2 fix is broken — abort "
        "would still be swallowed by Cellpose's logging chain."
    )
    assert caught_by_base_exception


def test_training_aborted_error_caught_by_explicit_clause():
    """Explicit `except TrainingAbortedError` catches it (most common use)."""
    caught = False
    try:
        raise TrainingAbortedError("cancelled")
    except TrainingAbortedError:
        caught = True
    assert caught


def test_training_aborted_error_caught_by_base_exception_explicitly():
    """A defensive top-level `except BaseException` also catches it."""
    caught = False
    try:
        raise TrainingAbortedError("cancelled")
    except BaseException:
        caught = True
    assert caught


# ---------------------------------------------------------------------------
# cellpose_trainer's narrowed except no longer swallows the abort
# ---------------------------------------------------------------------------


def test_epoch_log_handler_does_not_swallow_training_aborted_error():
    """The bare `except Exception` was the original swallow.  Now it's
    narrowed to specific parse errors.  TrainingAbortedError must
    propagate."""
    import logging

    aborts_propagated = []

    def _callback(epoch, n_epochs, train_loss, test_loss):
        raise TrainingAbortedError("user cancelled mid-epoch")

    handler = _EpochLogHandler(n_epochs=10, callback=_callback)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="5, train_loss=0.5, test_loss=0.3",
        args=(),
        exc_info=None,
    )

    with pytest.raises(TrainingAbortedError):
        handler.emit(record)


def test_epoch_log_handler_still_swallows_parse_errors():
    """Defensive parse-error handling still works (non-fatal)."""
    import logging

    def _bad_callback(epoch, n_epochs, train_loss, test_loss):
        raise TypeError("type error from callback — non-fatal")

    handler = _EpochLogHandler(n_epochs=10, callback=_bad_callback)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="5, train_loss=0.5, test_loss=0.3",
        args=(),
        exc_info=None,
    )

    # Should NOT raise — TypeError is in the narrowed-except list.
    handler.emit(record)


# ---------------------------------------------------------------------------
# F3.1 archive / restore / delete helpers
# ---------------------------------------------------------------------------


def test_archive_with_no_prior_state_returns_none(tmp_path):
    """If save_path has nothing to archive, returns None (fresh round)."""
    save_path = tmp_path / "round_00"
    save_path.mkdir()

    result = _archive_prior_training_state(save_path, "model_a")
    assert result is None


def test_archive_moves_cellpose_subdir(tmp_path):
    """The Cellpose models/ subdir gets moved to .pre_train_archive."""
    save_path = tmp_path / "round_00"
    save_path.mkdir()
    cellpose_subdir = save_path / "models"
    cellpose_subdir.mkdir()
    (cellpose_subdir / "weights.pth").write_text("PRIOR_WEIGHTS")

    archive_path = _archive_prior_training_state(save_path, "model_a")

    assert archive_path is not None
    assert archive_path.is_dir()
    assert (archive_path / "models" / "weights.pth").is_file()
    assert (archive_path / "models" / "weights.pth").read_text() == "PRIOR_WEIGHTS"
    # The original models/ should NOT exist anymore (moved, not copied).
    assert not cellpose_subdir.exists()


def test_archive_moves_companion_files(tmp_path):
    """Companion files matching {model_name}* at bundle root also archived."""
    save_path = tmp_path / "round_00"
    save_path.mkdir()
    (save_path / "model_a_train_loss.npy").write_text("LOSSES")
    (save_path / "model_a_training_config.json").write_text("{}")
    (save_path / "manifest.json").write_text("{}")  # NOT a companion — should stay

    archive_path = _archive_prior_training_state(save_path, "model_a")

    assert archive_path is not None
    assert (archive_path / "companions" / "model_a_train_loss.npy").is_file()
    assert (archive_path / "companions" / "model_a_training_config.json").is_file()
    # manifest.json (NOT prefixed with model_name) stays in save_path
    assert (save_path / "manifest.json").is_file()
    # Companions were MOVED out of save_path
    assert not (save_path / "model_a_train_loss.npy").exists()


def test_delete_archive_removes_directory(tmp_path):
    """_delete_archive cleans up the archive directory (success path)."""
    save_path = tmp_path / "round_00"
    save_path.mkdir()
    (save_path / "models").mkdir()
    (save_path / "models" / "w.pth").write_text("old")

    archive_path = _archive_prior_training_state(save_path, "model_a")
    assert archive_path.exists()

    _delete_archive(archive_path)
    assert not archive_path.exists()


def test_delete_archive_handles_none(tmp_path):
    """_delete_archive is a no-op when archive_path is None."""
    _delete_archive(None)  # Must not raise


def test_restore_prior_training_state_recovers_models(tmp_path):
    """_restore_prior_training_state moves archived state back."""
    save_path = tmp_path / "round_00"
    save_path.mkdir()
    (save_path / "models").mkdir()
    (save_path / "models" / "weights.pth").write_text("PRIOR")
    (save_path / "model_a_train_loss.npy").write_text("LOSSES")

    archive_path = _archive_prior_training_state(save_path, "model_a")
    # Simulate training partially writing then failing
    (save_path / "models").mkdir(exist_ok=True)
    (save_path / "models" / "partial.pth").write_text("PARTIAL")

    _restore_prior_training_state(save_path, archive_path)

    # Prior weights restored
    assert (save_path / "models" / "weights.pth").is_file()
    assert (save_path / "models" / "weights.pth").read_text() == "PRIOR"
    # Partial output from failed training was cleared
    assert not (save_path / "models" / "partial.pth").exists()
    # Companion restored
    assert (save_path / "model_a_train_loss.npy").is_file()
    assert (save_path / "model_a_train_loss.npy").read_text() == "LOSSES"
    # Archive cleaned up
    assert not archive_path.exists()


def test_archive_restore_roundtrip_preserves_content(tmp_path):
    """End-to-end: archive → restore returns identical content."""
    save_path = tmp_path / "round_00"
    save_path.mkdir()
    (save_path / "models").mkdir()
    (save_path / "models" / "w.pth").write_bytes(b"\x00\x01\x02BINARY")
    (save_path / "models" / "extras.json").write_text('{"a": 1}')
    (save_path / "model_x_loss.npy").write_bytes(b"\x03\x04")

    archive_path = _archive_prior_training_state(save_path, "model_x")
    _restore_prior_training_state(save_path, archive_path)

    assert (save_path / "models" / "w.pth").read_bytes() == b"\x00\x01\x02BINARY"
    assert (save_path / "models" / "extras.json").read_text() == '{"a": 1}'
    assert (save_path / "model_x_loss.npy").read_bytes() == b"\x03\x04"


# ---------------------------------------------------------------------------
# Source-inspection tests for the controller wiring (defends against
# someone removing the F3.1 / cancel-UX wiring inadvertently)
# ---------------------------------------------------------------------------


def test_cancel_training_updates_button_label():
    """cancel_training() must change button text to 'Cancelling...'"""
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )

    source = inspect.getsource(TrainingAnnotationController.cancel_training)
    assert "Cancelling" in source, (
        "cancel_training must set button text to 'Cancelling...' so the "
        "user knows the cancel was received and is in progress."
    )
    assert "setEnabled(False)" in source, (
        "cancel_training must disable the button to prevent re-click."
    )


def test_controller_imports_training_aborted_error_from_shared_module():
    """The controller no longer defines TrainingAbortedError locally —
    it imports from packages.epicycle_training.exceptions so the
    BaseException base is honored throughout the codebase."""
    import inspect
    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as ctrl_mod,
    )

    source = inspect.getsource(ctrl_mod)
    # Local class definition removed.
    assert "class TrainingAbortedError(" not in source, (
        "TrainingAbortedError must not be defined locally — it lives in "
        "packages/epicycle_training/exceptions.py"
    )
    # Import statement present.
    assert (
        "from packages.epicycle_training.exceptions import TrainingAbortedError"
        in source
    ) or (
        "from .exceptions import TrainingAbortedError" in source
    ) or (
        "epicycle_training.exceptions" in source
    ), "Controller must import TrainingAbortedError from the shared module"


def test_training_backend_adapter_uses_archive_pattern():
    """Source inspection: the adapter's training body must use
    _archive_prior_training_state + _delete_archive (success) +
    _restore_prior_training_state (failure)."""
    import inspect
    from packages.epicycle_training import training_backend_adapter as adapter

    source = inspect.getsource(adapter)
    # All three helpers must be defined.
    assert "def _archive_prior_training_state" in source
    assert "def _delete_archive" in source
    assert "def _restore_prior_training_state" in source
    # The training closure must use them.
    # Audit-fix-bundle (2026-05-27): the closure now calls
    # _archive_prior_training_state with log_warn_cb kwarg on a
    # multi-line form.  Match the function call by function-name only.
    assert "_archive_prior_training_state(" in source
    assert "_delete_archive(archive_path)" in source
    assert "_restore_prior_training_state(save_path, archive_path)" in source
    # And must catch BaseException (not just Exception) so the abort triggers restore.
    assert "except BaseException:" in source, (
        "F3.1 restore path must use `except BaseException` so "
        "TrainingAbortedError(BaseException) triggers the restore."
    )


def test_cellpose_trainer_narrowed_except():
    """The bare `except Exception: pass` in _EpochLogHandler.emit has
    been narrowed to specific parse errors.  Future regression of the
    bare except would re-introduce the swallow bug."""
    import inspect
    from packages.epicycle_training import cellpose_trainer

    source = inspect.getsource(cellpose_trainer._EpochLogHandler.emit)
    # The narrowed clause specifies multiple exception types.
    assert "except (" in source, (
        "_EpochLogHandler.emit must use a NARROWED except clause "
        "(e.g. `except (re.error, ValueError, TypeError, AttributeError):`)"
    )
    # And the comment block calls out the rationale.
    assert "TrainingAbortedError" in source, (
        "_EpochLogHandler.emit's docstring/comments should mention "
        "TrainingAbortedError so future maintainers understand why the "
        "except is narrowed."
    )


# ===========================================================================
# Audit-fix-bundle tests (2026-05-27) — covers all 7 findings from the
# 3-reviewer audit on commit 5049249.
# ===========================================================================


# ---------------------------------------------------------------------------
# Fix #1 — cancel_training gated on training-running
# ---------------------------------------------------------------------------


def test_cancel_training_gated_on_running_state():
    """cancel_training must early-return when no training is in flight,
    and must NOT mutate button state in that case."""
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )

    source = inspect.getsource(TrainingAnnotationController.cancel_training)
    # Must check sync_in_flight + thread_running before mutating state.
    assert "_sync_training_in_flight" in source, (
        "cancel_training must check `_sync_training_in_flight` so sync-"
        "mode (test/headless) cancels are recognized."
    )
    assert "isRunning()" in source or "isRunning" in source, (
        "cancel_training must check the QThread `isRunning()` so stale "
        "invocations after `_on_training_finished` don't mutate the "
        "button (leaving it stuck at 'Cancelling...' disabled)."
    )
    # Must early-return BEFORE mutating button.
    src_lines = source.splitlines()
    # Find the gate check + early-return + the button mutation.  The
    # gate check is the `if not (sync_in_flight or thread_running):`
    # line; the button mutation is "Cancelling..." text.
    gate_idx = next(
        (i for i, l in enumerate(src_lines)
         if "sync_in_flight" in l and "thread_running" in l and "if not" in l),
        -1,
    )
    # Match the ACTUAL setText call, not the docstring mention of
    # "Cancelling..." (which appears earlier in the function).
    btn_idx = next(
        (i for i, l in enumerate(src_lines)
         if 'setText("Cancelling...")' in l),
        -1,
    )
    assert gate_idx != -1, (
        "no gate check `if not (sync_in_flight or thread_running):` found"
    )
    assert btn_idx != -1, "no 'Cancelling...' button mutation found"
    assert gate_idx < btn_idx, (
        f"gate check at line {gate_idx} must be BEFORE button "
        f"mutation at line {btn_idx} — otherwise stale calls still "
        "mutate state."
    )
    # And there must be a `return` after the gate (within 12 lines — the
    # gate has a multi-line print() before the return).
    block_after_gate = "\n".join(src_lines[gate_idx : gate_idx + 14])
    assert "return" in block_after_gate, (
        "gate check must early-return when no training is in flight"
    )


def test_sync_training_in_flight_flag_setting():
    """The sync-fallback block must set + clear `_sync_training_in_flight`
    so cancel_training can detect sync-mode runs."""
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )

    # The flag-setting lives inside _train_model.  Use source-string
    # inspection of the whole module so we don't have to introspect
    # private helpers by name.
    import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as ctrl_mod
    source = inspect.getsource(ctrl_mod)
    assert "self._sync_training_in_flight = True" in source
    assert "self._sync_training_in_flight = False" in source
    # Clearing must be in a finally so a crash doesn't leave it stuck.
    assert "finally:" in source
    # Both lines together — locate the True-set and verify finally exists in
    # a nearby (within 30 lines) block.
    lines = source.splitlines()
    true_idx = next(
        i for i, l in enumerate(lines)
        if "self._sync_training_in_flight = True" in l
    )
    nearby = "\n".join(lines[true_idx : true_idx + 30])
    assert "finally:" in nearby
    assert "self._sync_training_in_flight = False" in nearby


# ---------------------------------------------------------------------------
# Fix #2 — sync fallback catches TrainingAbortedError before Exception
# ---------------------------------------------------------------------------


def test_sync_fallback_catches_training_aborted_error():
    """The _use_sync_training path must explicitly catch
    TrainingAbortedError BEFORE catching Exception, mirroring the
    _TrainingWorker.run pattern.  Otherwise BaseException-derived abort
    escapes unhandled in sync mode."""
    import inspect
    import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as ctrl_mod

    source = inspect.getsource(ctrl_mod)
    # Locate the TRAINING sync-fallback try block specifically.  2026-05-27:
    # `_use_sync_training` now also appears in `_confirm_revert_committed`
    # (edit-states branch), so we can't just grab the first occurrence —
    # anchor on the one whose following block calls `_run_training_work`.
    lines = source.splitlines()
    candidate_idxs = [
        i for i, l in enumerate(lines)
        if 'if getattr(self, "_use_sync_training"' in l
    ]
    sync_idx = None
    for idx in candidate_idxs:
        window = "\n".join(lines[idx: idx + 30])
        if "_run_training_work" in window:
            sync_idx = idx
            break
    assert sync_idx is not None, (
        "could not locate the training sync-fallback block "
        "(the _use_sync_training branch that calls _run_training_work)"
    )
    # The next lines must contain `except TrainingAbortedError:`
    # BEFORE `except Exception as exc:`.
    block = "\n".join(lines[sync_idx : sync_idx + 30])
    aborted_at = block.find("except TrainingAbortedError")
    exc_at = block.find("except Exception as exc")
    assert aborted_at != -1, (
        "sync-fallback block must explicitly catch TrainingAbortedError"
    )
    assert exc_at != -1, (
        "sync-fallback block must still catch Exception for non-abort failures"
    )
    assert aborted_at < exc_at, (
        "TrainingAbortedError catch must come BEFORE the Exception "
        "catch (Python tries clauses in order)"
    )


# ---------------------------------------------------------------------------
# Fix #3 — orphan archive scanner + restore
# ---------------------------------------------------------------------------


def test_scan_orphan_archives_empty_workspace(tmp_path):
    """No archives → empty result."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert scan_orphan_archives(workspace) == []


def test_scan_orphan_archives_handles_missing_path(tmp_path):
    """Non-existent workspace → empty result, no crash."""
    missing = tmp_path / "does_not_exist"
    assert scan_orphan_archives(missing) == []
    assert scan_orphan_archives(None) == []


def test_scan_orphan_archives_detects_orphans(tmp_path):
    """An orphan .pre_train_archive dir is found, save_path resolved.

    CRITICAL: archive layout (per real `_archive_prior_training_state`):
        save_path/                              # e.g. .../models/round_00/
        save_path.parent/.pre_train_archive/    # sibling of save_path
            {ts}_{save_path.name}_{model_name}/
                models/...
                companions/...

    So `archive_root.parent` is the SAVE_PATH PARENT, not save_path itself.
    Scanner must reconstruct save_path by parsing the dir name."""
    workspace = tmp_path / "workspace"
    save_path = workspace / "runs/myrun/lateral/models/round_00"
    save_path.mkdir(parents=True)
    # Archive at the REAL layout: sibling of save_path.
    archive_root = save_path.parent / ".pre_train_archive"
    archive_root.mkdir()
    orphan_dir = archive_root / "20260527T120000_round_00_weights"
    orphan_dir.mkdir()
    (orphan_dir / "models").mkdir()
    (orphan_dir / "models" / "weights.pth").write_text("PRIOR")

    orphans = scan_orphan_archives(workspace)
    assert len(orphans) == 1
    o = orphans[0]
    assert o["archive_path"] == orphan_dir
    # Critical: save_path must point to the ORIGINAL save_path, not its
    # parent.  Restore reads this and writes to save_path/models/.
    assert o["save_path"] == save_path, (
        f"scanner resolved save_path={o['save_path']} but expected "
        f"{save_path}.  Restore would write to the WRONG location."
    )
    assert o["ts"] == "20260527T120000"
    assert o["model_name_hint"] == "weights"
    # canonical models/ is empty → not present
    assert o["canonical_weights_present"] is False


def test_scan_orphan_archives_canonical_present(tmp_path):
    """When canonical save_path/models has files, canonical_present=True."""
    workspace = tmp_path / "workspace"
    save_path = workspace / "models/round_00"
    save_path.mkdir(parents=True)
    archive_root = save_path.parent / ".pre_train_archive"
    archive_root.mkdir()
    orphan_dir = archive_root / "20260527T120000_round_00_weights"
    orphan_dir.mkdir()
    # Canonical has weights too
    (save_path / "weights.pth").write_text("NEW")

    orphans = scan_orphan_archives(workspace)
    assert len(orphans) == 1
    assert orphans[0]["save_path"] == save_path
    assert orphans[0]["canonical_weights_present"] is True


def test_restore_orphan_archive_recovers(tmp_path):
    """restore_orphan_archive moves archived state back into save_path.

    Uses the REAL layout (archive at save_path.parent) so this catches
    the path-resolution bug the second-pass audit found."""
    save_path = tmp_path / "models" / "round_00"
    save_path.mkdir(parents=True)
    archive_root = save_path.parent / ".pre_train_archive"
    archive_root.mkdir()
    orphan_dir = archive_root / "20260527T120000_round_00_weights"
    orphan_dir.mkdir()
    (orphan_dir / "models").mkdir()
    (orphan_dir / "models" / "w.pth").write_text("PRIOR")

    orphan = {
        "archive_path": orphan_dir,
        "save_path": save_path,
        "ts": "20260527T120000",
        "model_name_hint": "weights",
        "canonical_weights_present": False,
    }
    ok = restore_orphan_archive(orphan)
    assert ok is True
    assert (save_path / "models" / "w.pth").is_file()
    assert (save_path / "models" / "w.pth").read_text() == "PRIOR"
    assert not orphan_dir.exists()


def test_orphan_scan_end_to_end_with_real_archive_helper(tmp_path):
    """End-to-end: call real _archive_prior_training_state, then
    scan_orphan_archives + restore_orphan_archive.  Verifies the
    full round-trip works with the actual layout, NOT a hand-built
    fake.  Catches path-resolution bugs the unit tests might miss."""
    workspace = tmp_path / "workspace"
    save_path = workspace / "runs" / "myrun" / "lateral" / "models" / "round_00"
    save_path.mkdir(parents=True)
    (save_path / "models").mkdir()
    (save_path / "models" / "weights.pth").write_bytes(b"PRIOR_BINARY")
    (save_path / "weights_train_loss.npy").write_bytes(b"\x01\x02\x03")

    # 1. Use the REAL helper to archive.
    archive_path = _archive_prior_training_state(save_path, "weights")
    assert archive_path is not None
    assert archive_path.is_dir()
    # The canonical save_path/models is now gone (archived).
    assert not (save_path / "models" / "weights.pth").exists()

    # 2. Scan from the workspace root.
    orphans = scan_orphan_archives(workspace)
    assert len(orphans) == 1, (
        f"expected exactly 1 orphan; got {len(orphans)}: "
        f"{[str(o['archive_path']) for o in orphans]}"
    )
    o = orphans[0]
    # 3. CRITICAL: save_path must point back to the ORIGINAL save_path.
    assert o["save_path"] == save_path, (
        f"scanner resolved save_path={o['save_path']} but real save_path "
        f"is {save_path}.  Restore would write prior weights to the "
        f"WRONG directory (auto-recovery silently broken)."
    )
    # Canonical weights are gone → present=False → eligible for auto-restore.
    assert o["canonical_weights_present"] is False
    assert o["model_name_hint"] == "weights"

    # 4. Restore via the public helper.
    ok = restore_orphan_archive(o)
    assert ok is True

    # 5. Prior weights and companions back at the ORIGINAL locations.
    assert (save_path / "models" / "weights.pth").is_file()
    assert (save_path / "models" / "weights.pth").read_bytes() == b"PRIOR_BINARY"
    assert (save_path / "weights_train_loss.npy").is_file()
    assert (save_path / "weights_train_loss.npy").read_bytes() == b"\x01\x02\x03"

    # 6. Archive cleaned up.
    assert not archive_path.exists()


def test_detect_stale_marker_wires_orphan_scan():
    """The controller's _detect_stale_training_marker must call the
    orphan scanner — otherwise crashes silently leak archives."""
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )

    source = inspect.getsource(
        TrainingAnnotationController._detect_stale_training_marker
    )
    assert "scan_orphan_archives" in source, (
        "_detect_stale_training_marker must call scan_orphan_archives "
        "so crashed-mid-train sessions get their prior weights back"
    )
    assert "restore_orphan_archive" in source, (
        "must also wire restore_orphan_archive for the auto-recovery path"
    )


# ---------------------------------------------------------------------------
# Fix #4 — archive fallback surfaces to UI log
# ---------------------------------------------------------------------------


def test_archive_helper_accepts_log_warn_cb():
    """_archive_prior_training_state must accept log_warn_cb kwarg."""
    import inspect
    sig = inspect.signature(_archive_prior_training_state)
    assert "log_warn_cb" in sig.parameters


def test_archive_fallback_calls_log_warn_cb(tmp_path, monkeypatch):
    """When shutil.move fails, the helper must call log_warn_cb with a
    message describing what got discarded."""
    import shutil

    save_path = tmp_path / "round_00"
    save_path.mkdir()
    cellpose_subdir = save_path / "models"
    cellpose_subdir.mkdir()
    (cellpose_subdir / "weights.pth").write_text("PRIOR")

    # Force shutil.move to fail to simulate a Windows file-lock.
    def _failing_move(src, dst):
        raise PermissionError("simulated file lock")

    # shutil is imported INSIDE the helper, which binds to the module-
    # level shutil singleton.  Monkeypatching shutil.move globally is
    # the right surface.
    monkeypatch.setattr(shutil, "move", _failing_move)

    received_warnings = []
    _archive_prior_training_state(
        save_path, "weights",
        log_warn_cb=received_warnings.append,
    )
    assert len(received_warnings) >= 1, (
        "log_warn_cb must be called when archive falls back to rmtree"
    )
    joined = " ".join(received_warnings)
    assert "DISCARDED" in joined or "discarded" in joined or "could not be archived" in joined


def test_training_worker_emits_log_warn_signal():
    """_TrainingWorker must have a log_warn Signal that gets forwarded
    to the training closure as log_warn_cb."""
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        _TrainingWorker,
    )

    source = inspect.getsource(_TrainingWorker)
    assert "log_warn = Signal(str)" in source, (
        "_TrainingWorker must declare a log_warn signal so warnings "
        "from the worker thread can be safely surfaced to the UI"
    )
    assert "log_warn_cb=self.log_warn.emit" in source, (
        "_TrainingWorker.run must pass log_warn=self.log_warn.emit to "
        "the train_fn so the closure can surface warnings"
    )


def test_run_training_work_forwards_log_warn_cb():
    """_run_training_work must accept log_warn_cb and forward to
    training_manager.run_training."""
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )

    sig = inspect.signature(TrainingAnnotationController._run_training_work)
    assert "log_warn_cb" in sig.parameters, (
        "_run_training_work must accept log_warn_cb"
    )
    source = inspect.getsource(TrainingAnnotationController._run_training_work)
    assert "log_warn_cb=log_warn_cb" in source, (
        "_run_training_work must forward log_warn_cb to "
        "training_manager.run_training"
    )


# ---------------------------------------------------------------------------
# Fix #5 — _delete_archive gated on weights_lifted (new layout) or
# canonical-file-exists (legacy)
# ---------------------------------------------------------------------------


def test_delete_archive_gated_on_lift_success():
    """The training closure must NOT call _delete_archive when
    weights_lifted=False — that would orphan both old + new."""
    import inspect
    from packages.epicycle_training import training_backend_adapter as adapter

    source = inspect.getsource(adapter)
    # The success-path delete must be gated.
    # Find the `_delete_archive(archive_path)` call sites; at least one
    # must be conditional on weights_lifted / _new_weights_present.
    assert "_new_weights_present" in source, (
        "training closure must compute `_new_weights_present` before "
        "deciding whether to delete the archive"
    )
    # And the new-layout branch checks `weights_lifted`.
    assert "_new_weights_present = bool(weights_lifted)" in source, (
        "new-layout branch must gate on weights_lifted"
    )
    # The legacy branch checks the checkpoint_path file existence.
    assert "Path(checkpoint_path).is_file()" in source, (
        "legacy branch must gate on the actual checkpoint file existing"
    )


# ---------------------------------------------------------------------------
# Fix #6 — INTEGRATION test for archive→restore via the closure
# ---------------------------------------------------------------------------


def _build_stub_training_samples(tmp_path):
    """Create a minimal training sample TIFF pair for integration tests."""
    import numpy as np
    import tifffile

    img_path = tmp_path / "img.tif"
    mask_path = tmp_path / "mask.tif"
    img = np.zeros((32, 32), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint16)
    tifffile.imwrite(str(img_path), img)
    tifffile.imwrite(str(mask_path), mask)
    return [{
        "stack_id": "stack1",
        "view": "XY",
        "slice_index": 0,
        "img_uri": str(img_path),
        "mask_uri": str(mask_path),
        "slice_label": "stack1_XY_z0000",
    }]


def test_integration_archive_restored_on_abort(tmp_path, monkeypatch):
    """End-to-end: stub CellposeTrainer.fit raises TrainingAbortedError,
    backend closure restores the archive, prior weights are back at
    the canonical location."""
    backend = build_epicycle_training_backend()
    if backend is None:
        pytest.skip("cellpose not installed; backend factory returned None")

    save_path = tmp_path / "round_00"
    save_path.mkdir()
    # Seed prior weights at canonical location.
    (save_path / "models").mkdir()
    (save_path / "models" / "weights.pth").write_text("PRIOR_WEIGHTS")
    # Also a companion file.
    (save_path / "weights_train_loss.npy").write_text("PRIOR_LOSS")

    # Stub the CellposeTrainer so fit() raises mid-train.
    class _StubTrainer:
        train_loss = None
        val_loss = None
        model_path = None
        def __init__(self, *a, **kw): pass
        def fit(self, **kw):
            # Simulate partial write before abort.
            (save_path / "models").mkdir(exist_ok=True)
            (save_path / "models" / "partial.pth").write_text("PARTIAL")
            raise TrainingAbortedError("user cancelled")

    monkeypatch.setattr(
        "packages.epicycle_training.training_backend_adapter.CellposeTrainer",
        _StubTrainer,
    )

    samples = _build_stub_training_samples(tmp_path)
    config = {"epochs": 1, "channel1": "0: gray", "channel2": "0: none"}

    with pytest.raises(TrainingAbortedError):
        backend(
            round_id="round_00",
            config=config,
            working_dir=str(tmp_path),
            training_samples=samples,
            view_type="XY",
            save_path_override=str(save_path),
            weights_filename="weights",
        )

    # Prior weights restored at canonical.
    assert (save_path / "models" / "weights.pth").is_file()
    assert (save_path / "models" / "weights.pth").read_text() == "PRIOR_WEIGHTS"
    # Partial output from the failed run was cleared.
    assert not (save_path / "models" / "partial.pth").exists()
    # Companion restored.
    assert (save_path / "weights_train_loss.npy").is_file()
    assert (save_path / "weights_train_loss.npy").read_text() == "PRIOR_LOSS"
    # Archive cleaned up.
    archive_root = save_path.parent / ".pre_train_archive"
    if archive_root.exists():
        # Should be empty or gone entirely.
        assert not any(archive_root.iterdir())


def test_integration_archive_deleted_on_success_only_when_lifted(tmp_path, monkeypatch):
    """End-to-end: stub fit succeeds, but stub lift returns
    weights_lifted=False.  Archive must NOT be deleted — user can recover."""
    import numpy as np

    backend = build_epicycle_training_backend()
    if backend is None:
        pytest.skip("cellpose not installed")

    save_path = tmp_path / "round_00"
    save_path.mkdir()
    (save_path / "models").mkdir()
    (save_path / "models" / "weights.pth").write_text("PRIOR")

    class _StubTrainer:
        train_loss = np.array([0.5])
        val_loss = None
        model_path = save_path / "models" / "weights"
        def __init__(self, *a, **kw): pass
        def fit(self, **kw):
            # Simulate cellpose writing under models/
            (save_path / "models").mkdir(exist_ok=True)
            (save_path / "models" / "weights").write_text("NEW_WEIGHTS")

    # Stub lift to claim it didn't lift.
    def _stub_lift(*, save_path, weights_filename, log_fn):
        return {"weights_lifted": False, "skipped_reason": "dst_exists"}

    monkeypatch.setattr(
        "packages.epicycle_training.training_backend_adapter.CellposeTrainer",
        _StubTrainer,
    )
    monkeypatch.setattr(
        "packages.epicycle_training.cellpose_lift.lift_cellpose_output",
        _stub_lift,
    )

    samples = _build_stub_training_samples(tmp_path)
    config = {"epochs": 1, "channel1": "0: gray", "channel2": "0: none"}

    result = backend(
        round_id="round_00",
        config=config,
        working_dir=str(tmp_path),
        training_samples=samples,
        view_type="XY",
        save_path_override=str(save_path),
        weights_filename="weights",
    )
    assert result["weights_lifted"] is False
    # Archive should still exist (Fix #5: only delete if lifted).
    archive_root = save_path.parent / ".pre_train_archive"
    assert archive_root.exists()
    children = list(archive_root.iterdir())
    assert len(children) >= 1, (
        "archive must be preserved when weights_lifted=False — "
        "otherwise user loses both old (deleted) and new (orphaned)"
    )

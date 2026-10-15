"""Tests for the QC worker patch-name guard (2026-05-31, fix/step3-qc-worker-patch-guard).

Background: `_QCFeatureWorker` runs on a QThread for minutes per patch on
large volumes.  If the user clicks a different patch while extraction is in
flight, the worker keeps running on the original patch and eventually fires
its `finished` signal.  Before this fix, `_on_qc_extract_finished` did
``self._qc_features_df = df`` unconditionally — overwriting the active (new)
patch's df with the old patch's features.  Navigator and QC plot would
silently show old-patch features under new-patch label ids.

The fix is a single early-return guard at the top of
`_on_qc_extract_finished` that compares the worker's `patch_name` to
`self._qc_view_patch_name` (the currently-active patch).  If they differ:
- The on-disk CSV is preserved (it was written by the worker for the
  ORIGINAL patch — correct).
- The in-memory df for the active patch is NOT touched.
- A `[QC-EXTRACT] STALE FINISH ignored` line is printed to stdout, plus a
  UI-log line telling the user to re-click the patch to view its features.
"""
from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


def test_on_qc_extract_finished_has_patch_name_guard():
    """The guard must compare the worker's patch_name against the active
    patch (`_qc_view_patch_name`) and early-return on mismatch.
    Without this, a cross-patch worker contamination silently corrupts the
    in-memory df."""
    src = inspect.getsource(
        FusionReconstructionController._on_qc_extract_finished
    )
    assert "_qc_view_patch_name" in src, (
        "Guard must read self._qc_view_patch_name to compare against the "
        "worker's patch_name."
    )
    # Must early-return on mismatch.  Look for the return statement
    # immediately after the comparison block.
    assert "patch_name != active_patch" in src, (
        "Guard must use the patch_name != active_patch comparison so the "
        "intent is clear in the source."
    )
    # Must have a return statement — the guard's purpose is to bail before
    # the existing logic runs.
    assert src.count("return") >= 1, "Guard must early-return on mismatch."


def test_guard_emits_user_visible_log_line():
    """The guard must also emit a UI-log line via `_append_log` so the
    user sees a record in the napari log panel — not just stdout.  Tells
    them what happened + how to recover (re-click the abandoned patch).
    """
    src = inspect.getsource(
        FusionReconstructionController._on_qc_extract_finished
    )
    assert "_append_log" in src, (
        "Guard must emit a UI-log line via self._append_log so the user "
        "sees the staleness notification in the napari log panel."
    )
    assert "Re-click" in src, (
        "User-facing message should suggest re-clicking the abandoned "
        "patch to recover its features."
    )


def test_worker_emits_qc_extract_console_markers():
    """START + DONE markers emit a `[QC-EXTRACT]` stdout line with
    patch_name so the user can correlate console output to which patch
    was actually processed.  Critical when the guard fires — the stdout
    trail proves which patch the worker was on.

    2026-06-07 (feature/step3-extract-subprocess): the START/DONE markers
    now live in the relocated pure compute function
    `_qc_extract_core.run_full_qc_extraction` (which runs in the
    subprocess AND in the in-process fallback worker), while the ERROR
    marker still lives in the worker's `run()` wrapper.  The markers are
    unchanged — only WHERE they live moved."""
    import inspect as _inspect
    from napari_epicycle.fusion_reconstruction import controller as _ctrl_mod
    from napari_epicycle.fusion_reconstruction import _qc_extract_core as _core

    # START + DONE markers live in the relocated core compute function.
    core_src = _inspect.getsource(_core.run_full_qc_extraction)
    assert "[QC-EXTRACT] Worker START" in core_src
    assert "[QC-EXTRACT] Worker DONE" in core_src
    # Each marker includes patch_name so the cross-patch contamination
    # case is visible.
    assert "patch_name" in core_src

    # ERROR marker still lives in the worker's run() wrapper.
    worker_src = _inspect.getsource(_ctrl_mod._QCFeatureWorker.run)
    assert "[QC-EXTRACT] Worker ERROR" in worker_src
    assert "self._patch_name" in worker_src


# ---------------------------------------------------------------------------
# Behavioral test — the source-only tests above can pass on a logically
# broken guard (e.g. inverted comparison).  Audit-found gap (2026-05-31).
# This test constructs a minimal controller fake with just the fields
# the guard reads/writes, calls the unbound method directly, and asserts
# the in-memory df is NOT overwritten on mismatch.
# ---------------------------------------------------------------------------


class _FakeExtractBtn:
    def __init__(self):
        self.enabled = False
        self.text = "Extracting..."

    def setEnabled(self, b):
        self.enabled = b

    def setText(self, t):
        self.text = t


class _FakeUI:
    def __init__(self):
        self.extract_features_btn = _FakeExtractBtn()


def _build_minimal_controller_fake(active_patch: str, existing_df):
    """Construct a bare controller object with just the attributes the
    guard reads/writes.  We bypass __init__ via __new__ so we don't
    need Qt / napari / state-manager setup."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    f = FusionReconstructionController.__new__(FusionReconstructionController)
    f._qc_view_patch_name = active_patch
    f._qc_features_df = existing_df
    f.ui = _FakeUI()
    f._log_messages = []
    f._append_log = lambda msg: f._log_messages.append(msg)
    return f


def test_guard_blocks_overwrite_on_patch_mismatch(capsys):
    """The crucial behavioral assertion: when the worker's patch_name
    differs from the active patch, _on_qc_extract_finished must NOT
    overwrite _qc_features_df."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    # Sentinel: a df-stand-in unique enough we can verify it survived.
    existing_df = {"sentinel": "PATCH_B_DF"}
    incoming_df = {"sentinel": "PATCH_A_DF"}

    f = _build_minimal_controller_fake(active_patch="patch_B", existing_df=existing_df)

    # Call the unbound method directly.  Worker says it finished for
    # patch_A; active patch is patch_B.
    FusionReconstructionController._on_qc_extract_finished(f, incoming_df, "patch_A")

    # The active patch's df MUST be untouched.
    assert f._qc_features_df is existing_df
    assert f._qc_features_df["sentinel"] == "PATCH_B_DF"
    # The UI-log line must have been appended so the user sees the
    # staleness notice.
    assert any("Worker finished for 'patch_A'" in m for m in f._log_messages)
    # The Extract button must be re-enabled (audit-found UX bug —
    # otherwise it would stay stuck on "Extracting...").
    assert f.ui.extract_features_btn.enabled is True
    assert f.ui.extract_features_btn.text == "Extract"
    # And the [QC-EXTRACT] STALE FINISH stdout line must have been
    # printed.
    captured = capsys.readouterr()
    assert "[QC-EXTRACT] STALE FINISH" in captured.out
    assert "patch_A" in captured.out
    assert "patch_B" in captured.out


def test_guard_lets_matching_patch_through_to_normal_handling(capsys):
    """When patch_name matches the active patch, the guard must NOT fire
    — the normal extraction-finished logic should run.  We can't easily
    exercise the FULL normal path without pandas + state_manager, but we
    can assert the guard didn't print STALE FINISH and didn't re-enable
    the button via the guard's early-return path."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    f = _build_minimal_controller_fake(active_patch="patch_A", existing_df=None)

    # Call directly — normal path will fail past the guard because our
    # fake doesn't have the full controller state (pandas, state_manager,
    # qc_features helpers).  But we only need to verify the guard didn't
    # short-circuit: capture stdout and assert no STALE FINISH log.
    try:
        FusionReconstructionController._on_qc_extract_finished(f, {"x": 1}, "patch_A")
    except Exception:
        # Expected — normal post-guard logic depends on real state we
        # haven't faked.  We don't care about the failure; we care that
        # the GUARD path didn't run.
        pass

    captured = capsys.readouterr()
    assert "[QC-EXTRACT] STALE FINISH" not in captured.out

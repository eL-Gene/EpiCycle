"""Reusable _close_3d_curation_annotator() helper.

The canonical Step-3 annotator teardown — dock.close() (→ closeEvent →
_close_cleanup) + optionally _clear_qc_view_layers + _clear_curation_layers +
llm.clear_step("step3") — was inlined in on_step_exit and two sibling close
sites.  It is extracted into one reusable helper (the future run-switch will
call it; a run-switch is close + relaunch, not the in-place patch-switch).

The helper's dock-close block is fully try/except-guarded (the launcher import
+ _find_dock_by_title are both swallowed), so these tests exercise the
clear-layer behaviour with a dummy viewer without needing Qt / the launcher.
"""

from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


class _LLM:
    def __init__(self):
        self.cleared = []

    def clear_step(self, step):
        self.cleared.append(step)


def _make_controller():
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c.viewer = None  # _find_dock_by_title(None, ...) → swallowed by try/except
    c._calls = []
    c._clear_qc_view_layers = lambda: c._calls.append("clear_qc")
    c._clear_curation_layers = lambda: c._calls.append("clear_curation")
    c.layer_lifecycle_manager = _LLM()
    return c


def test_close_helper_clear_layers_true_runs_full_sequence():
    c = _make_controller()
    c._close_3d_curation_annotator()  # default clear_layers=True
    assert c._calls == ["clear_qc", "clear_curation"], (
        "clear_layers=True must run both layer clears in order"
    )
    assert c.layer_lifecycle_manager.cleared == ["step3"], (
        "clear_layers=True must call llm.clear_step('step3')"
    )


def test_close_helper_clear_layers_false_is_close_only():
    c = _make_controller()
    c._close_3d_curation_annotator(clear_layers=False)
    assert c._calls == [], "clear_layers=False must NOT run the layer clears"
    assert c.layer_lifecycle_manager.cleared == [], (
        "clear_layers=False must NOT call llm.clear_step"
    )


def test_close_helper_safe_without_llm():
    """No layer_lifecycle_manager → the clear_step step is skipped, no raise."""
    c = _make_controller()
    c.layer_lifecycle_manager = None
    c._close_3d_curation_annotator()
    assert c._calls == ["clear_qc", "clear_curation"]


def test_close_helper_layer_clear_exceptions_are_swallowed():
    """A clear raising must not abort the teardown (matches the prior inlined
    try/except per-clear)."""
    c = _make_controller()

    def _boom():
        raise RuntimeError("clear failed")

    c._clear_qc_view_layers = _boom
    # Must not raise; clear_curation + clear_step still run.
    c._close_3d_curation_annotator()
    assert c.layer_lifecycle_manager.cleared == ["step3"]


# ---------------------------------------------------------------------------
# Source: the three close sites delegate to the helper.
# ---------------------------------------------------------------------------


def test_on_step_exit_delegates_to_helper():
    src = inspect.getsource(FusionReconstructionController.on_step_exit)
    assert "_close_3d_curation_annotator()" in src, (
        "on_step_exit must delegate to the reusable helper"
    )
    # The inlined dock.close()/clear sequence must be gone from on_step_exit.
    assert "_find_dock_by_title" not in src, (
        "on_step_exit must no longer inline the dock-close"
    )


def test_external_removal_uses_close_only_helper():
    src = inspect.getsource(
        FusionReconstructionController._on_layer_externally_removed
    )
    assert "_close_3d_curation_annotator(clear_layers=False)" in src, (
        "_on_layer_externally_removed must close-only via the helper"
    )
    assert "_find_dock_by_title" not in src


def test_view_reconstruction_uses_close_only_helper():
    # A2: the foot's View and Session's eye share ``_view_patch``
    assert "_view_patch(" in inspect.getsource(
        FusionReconstructionController._view_reconstruction)
    src = inspect.getsource(FusionReconstructionController._view_patch)
    assert "_close_3d_curation_annotator(clear_layers=False)" in src, (
        "_view_reconstruction must close-only via the helper (its own "
        "context-clears stay inline)"
    )
    # Its own context clears must remain.
    assert 'clear_context("step3/view")' in src
    assert "_clear_curation_layers()" in src
    assert "_clear_qc_view_layers()" in src

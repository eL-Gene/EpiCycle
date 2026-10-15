"""D2.5 — Lossy-downsample-load warning gate logic.

Pinned contract: ``_should_warn_lossy_downsample(patch_name)`` returns
True iff loading this patch with workspace downsample_mode=True would
be a lossy mode-switch — i.e., the patch was last saved at full-res,
has actual curation work, AND the user hasn't already dismissed the
warning for this patch.

The modal itself (``_show_lossy_downsample_warning``) is Qt-dependent
and verified manually; the gate logic that decides WHEN to show is
tested here without Qt.
"""

from unittest.mock import MagicMock

import pytest


def _make_controller(
    *,
    workspace_downsample_mode: bool = False,
    patch_factor: int = 1,
    patch_status: str = "unprocessed",
    patch_warning_shown: bool = False,
    patch_name: str = "patch_a",
):
    """Build a controller via object.__new__ with state_manager mocks
    that return the values for each gate-decision input."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.state_manager = MagicMock()
    ctrl.state_manager.get_downsample_mode = MagicMock(
        return_value=workspace_downsample_mode
    )
    ctrl.state_manager.get_patch_downsample_factor = MagicMock(
        return_value=patch_factor
    )
    ctrl.state_manager.get_patch_status = MagicMock(return_value=patch_status)
    ctrl.state_manager.get_patch_downsample_warning_shown = MagicMock(
        return_value=patch_warning_shown
    )
    return ctrl, patch_name


# ---------------------------------------------------------------------------
# Should fire — all conditions met
# ---------------------------------------------------------------------------


class TestWarningFires:
    """Warning fires when ALL of these are true:
    - workspace toggle ON
    - patch's last load was full-res (factor=1)
    - patch has actual curation (status in {curating, accepted, finalized})
    - warning not already dismissed for this patch."""

    def test_curating_status_fires(self):
        ctrl, name = _make_controller(
            workspace_downsample_mode=True,
            patch_factor=1,
            patch_status="curating",
            patch_warning_shown=False,
        )
        assert ctrl._should_warn_lossy_downsample(name) is True

    def test_accepted_status_fires(self):
        ctrl, name = _make_controller(
            workspace_downsample_mode=True,
            patch_factor=1,
            patch_status="accepted",
            patch_warning_shown=False,
        )
        assert ctrl._should_warn_lossy_downsample(name) is True

    def test_finalized_status_fires(self):
        ctrl, name = _make_controller(
            workspace_downsample_mode=True,
            patch_factor=1,
            patch_status="finalized",
            patch_warning_shown=False,
        )
        assert ctrl._should_warn_lossy_downsample(name) is True


# ---------------------------------------------------------------------------
# Should NOT fire — at least one condition not met
# ---------------------------------------------------------------------------


class TestWarningSuppressed:
    """Warning is suppressed when any precondition fails."""

    def test_workspace_toggle_off_skips(self):
        """If downsample_mode is off, no downsample → no lossy load."""
        ctrl, name = _make_controller(
            workspace_downsample_mode=False,
            patch_factor=1,
            patch_status="accepted",
            patch_warning_shown=False,
        )
        assert ctrl._should_warn_lossy_downsample(name) is False

    def test_patch_already_at_downsample_skips(self):
        """If factor is already 2, the user already accepted the
        trade-off in a previous load — no lossy mode-SWITCH."""
        ctrl, name = _make_controller(
            workspace_downsample_mode=True,
            patch_factor=2,
            patch_status="accepted",
            patch_warning_shown=False,
        )
        assert ctrl._should_warn_lossy_downsample(name) is False

    def test_fresh_patch_unprocessed_skips(self):
        """Fresh patches have no curation to lose."""
        ctrl, name = _make_controller(
            workspace_downsample_mode=True,
            patch_factor=1,
            patch_status="unprocessed",
            patch_warning_shown=False,
        )
        assert ctrl._should_warn_lossy_downsample(name) is False

    def test_predicted_status_skips(self):
        """Predicted but not yet curated — no precision to lose."""
        ctrl, name = _make_controller(
            workspace_downsample_mode=True,
            patch_factor=1,
            patch_status="predicted",
            patch_warning_shown=False,
        )
        assert ctrl._should_warn_lossy_downsample(name) is False

    def test_reconstructed_status_skips(self):
        """Reconstructed but not yet user-curated."""
        ctrl, name = _make_controller(
            workspace_downsample_mode=True,
            patch_factor=1,
            patch_status="reconstructed",
            patch_warning_shown=False,
        )
        assert ctrl._should_warn_lossy_downsample(name) is False

    def test_already_dismissed_skips(self):
        """User has already opted Continue on this patch's warning —
        don't nag again."""
        ctrl, name = _make_controller(
            workspace_downsample_mode=True,
            patch_factor=1,
            patch_status="accepted",
            patch_warning_shown=True,
        )
        assert ctrl._should_warn_lossy_downsample(name) is False


# ---------------------------------------------------------------------------
# Defensive — exceptions don't propagate
# ---------------------------------------------------------------------------


class TestGateRobustness:
    """Gate must never raise — defensively returns False on any state-
    manager error (load proceeds with workspace setting unchanged)."""

    def test_returns_false_on_state_manager_exception(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.state_manager = MagicMock()
        ctrl.state_manager.get_downsample_mode = MagicMock(
            side_effect=RuntimeError("simulated")
        )
        # Must not raise.
        assert ctrl._should_warn_lossy_downsample("any_patch") is False

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


# ---------------------------------------------------------------------------
# The words: what a half-size save really does to the mask
# ---------------------------------------------------------------------------


class _FakeBox:
    """Stands in for QMessageBox: records the texts, clicks Cancel."""
    Warning = 2
    AcceptRole = 0
    RejectRole = 1
    last = None

    def __init__(self, parent=None):
        self.texts = {}
        self.buttons = []
        _FakeBox.last = self

    def setIcon(self, icon):
        pass

    def setWindowTitle(self, text):
        self.texts["title"] = text

    def setText(self, text):
        self.texts["text"] = text

    def setInformativeText(self, text):
        self.texts["info"] = text

    def addButton(self, text, role):
        btn = object()
        self.buttons.append((text, role, btn))
        return btn

    def setDefaultButton(self, btn):
        pass

    def exec_(self):
        return 0

    def clickedButton(self):
        return self.buttons[-1][2]


class TestWarningWords:
    """The box tells the truth about a half-size save."""

    def _box(self, monkeypatch):
        import qtpy.QtWidgets as qw
        monkeypatch.setattr(qw, "QMessageBox", _FakeBox)
        ctrl, name = _make_controller()
        ctrl._append_log = lambda *_a: None
        assert ctrl._show_lossy_downsample_warning(name) is False
        return _FakeBox.last

    def test_no_false_promises(self, monkeypatch):
        box = self._box(monkeypatch)
        info = box.texts["info"]
        assert "Cell IDs and tier assignments are preserved" not in info
        assert "Saved canonical shape is preserved" not in info
        assert "sub-2-voxel" not in info

    def test_says_what_is_lost(self, monkeypatch):
        info = self._box(monkeypatch).texts["info"]
        assert "2-voxel steps" in info
        assert "thin gaps between neighbouring cells can close" in info
        assert ("small or very thin cells (a few voxels, or one voxel "
                "thick) can disappear") in info
        assert "8 voxels" not in info
        # the cells go at load, the boundaries move at save
        assert "On load, small or very thin cells" in info
        assert ("on save the mask is upsampled back to full size with "
                "boundaries in 2-voxel steps") in info
        assert "marked deleted in the table when you save" in info
        assert "Tier assignments of the remaining cells are kept" in info
        assert "run Batch Refine once with a Smooth step" in info
        assert "full resolution" in info
        assert "Settle" not in info

    def test_title_and_buttons_unchanged(self, monkeypatch):
        box = self._box(monkeypatch)
        assert box.texts["title"] == (
            "Switching to downsample for a curated patch")
        assert [b[0] for b in box.buttons] == [
            "Continue — downsample anyway",
            "Cancel — load full-res for this patch",
        ]
        assert "Choose Continue to downsample anyway" in box.texts["info"]


def test_checkbox_tooltip_is_honest():
    from napari_epicycle.fusion_reconstruction.ui import (
        DOWNSAMPLE_MODE_TOOLTIP,
    )
    tip = DOWNSAMPLE_MODE_TOOLTIP
    assert "preserved at full resolution" not in tip
    assert "upsampled back to full size" in tip
    assert "2-voxel steps" in tip
    assert ("small or very thin cells (a few voxels, or one voxel "
            "thick) can disappear") in tip
    assert "8 voxels" not in tip
    assert "on load small or very thin cells" in tip
    assert ("on save the mask is upsampled back to full size with "
            "boundaries in 2-voxel steps") in tip
    assert "Re-click the patch in the bucket below" in tip

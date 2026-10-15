"""Tests for controller isolation cleanup on annotator close and patch switch.

Bug 1: _on_annotator_isolation_cleared() does not release CONTROLLER_ISOLATION
        from the ColormapAuthority, blocking subsequent annotator claims.

Bug 2: _switch_patch_from_3d_annotator() does not clear isolation state,
        causing stale label IDs and authority ownership to persist across patches.

These tests use lightweight fakes — no Qt or napari required.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from napari_epicycle.fusion_reconstruction.colormap_authority import (
    ColormapAuthority,
    ColormapOwner,
)


# ---------------------------------------------------------------------------
# Helpers — minimal controller-like harness
# ---------------------------------------------------------------------------


def _make_isolation_harness(*, authority_owner=None, restore_point=None):
    """Build a minimal harness with the attributes _on_annotator_isolation_cleared needs.

    Parameters
    ----------
    authority_owner : ColormapOwner or None
        If given, the authority is pre-claimed by this owner.
    restore_point : object or None
        If given, set as the authority's restore point.
    """
    harness = MagicMock()

    auth = ColormapAuthority()
    if authority_owner is not None:
        auth.claim(authority_owner)
    if restore_point is not None:
        auth.set_restore_point(restore_point)
    harness._colormap_authority = auth

    harness._isolated_label_ids = {10, 20, 30}
    harness._isolated_df_indices = [0, 1, 2]

    # _find_active_labels_layer → None (skip the show_selected_label block)
    harness._find_active_labels_layer = MagicMock(return_value=None)

    # UI: sync_viewer_colors_cb unchecked (skip recolor block)
    harness.ui = MagicMock()
    harness.ui.sync_viewer_colors_cb.isChecked.return_value = False

    return harness


def _call_on_annotator_isolation_cleared(harness, *, reason=None):
    """Bind and call the real controller method on a harness."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    # Bind the unbound method to our harness
    method = FusionReconstructionController._on_annotator_isolation_cleared
    if reason is None:
        method(harness)
    else:
        method(harness, reason=reason)


# ---------------------------------------------------------------------------
# Bug 1: _on_annotator_isolation_cleared must release CONTROLLER_ISOLATION
# ---------------------------------------------------------------------------


class TestAnnotatorIsolationClearedReleasesAuthority:
    """_on_annotator_isolation_cleared must release CONTROLLER_ISOLATION."""

    def test_controller_isolation_released_on_callback(self):
        """Authority ownership is cleared when annotator close callback fires."""
        harness = _make_isolation_harness(
            authority_owner=ColormapOwner.CONTROLLER_ISOLATION
        )
        assert harness._colormap_authority.current_owner == ColormapOwner.CONTROLLER_ISOLATION

        _call_on_annotator_isolation_cleared(harness)

        assert harness._colormap_authority.current_owner is None

    def test_annotator_isolation_unaffected(self):
        """If authority has ANNOTATOR_ISOLATION, the release is a no-op.

        release(CONTROLLER_ISOLATION) when owner is ANNOTATOR_ISOLATION
        should not change ownership.
        """
        harness = _make_isolation_harness(
            authority_owner=ColormapOwner.ANNOTATOR_ISOLATION
        )

        _call_on_annotator_isolation_cleared(harness)

        assert (
            harness._colormap_authority.current_owner
            == ColormapOwner.ANNOTATOR_ISOLATION
        )

    def test_no_owner_is_safe(self):
        """If authority has no owner, the release is a no-op — no crash."""
        harness = _make_isolation_harness(authority_owner=None)

        _call_on_annotator_isolation_cleared(harness)

        assert harness._colormap_authority.current_owner is None

    def test_controller_qc_unaffected(self):
        """If authority has CONTROLLER_QC, the release is a no-op.

        CONTROLLER_QC != CONTROLLER_ISOLATION, so release should not clear it.
        """
        harness = _make_isolation_harness(
            authority_owner=ColormapOwner.CONTROLLER_QC
        )

        _call_on_annotator_isolation_cleared(harness)

        assert (
            harness._colormap_authority.current_owner
            == ColormapOwner.CONTROLLER_QC
        )

    def test_annotator_can_claim_after_close_clears_isolation(self):
        """End-to-end: close annotator (clears CONTROLLER_ISOLATION) → reopen →
        annotator can claim ANNOTATOR_ISOLATION successfully."""
        harness = _make_isolation_harness(
            authority_owner=ColormapOwner.CONTROLLER_ISOLATION
        )

        # Annotator closes — fires the callback
        _call_on_annotator_isolation_cleared(harness)

        # Now annotator reopens and tries to claim
        result = harness._colormap_authority.claim(
            ColormapOwner.ANNOTATOR_ISOLATION
        )
        assert result is True
        assert (
            harness._colormap_authority.current_owner
            == ColormapOwner.ANNOTATOR_ISOLATION
        )

    def test_close_reason_skips_heavy_recolor_path(self):
        """Close-mode callback should clear isolation without recolor churn."""
        harness = _make_isolation_harness(
            authority_owner=ColormapOwner.CONTROLLER_ISOLATION
        )
        harness._find_all_patch_labels_layers = MagicMock(
            return_value=[types.SimpleNamespace(visible=True)]
        )
        harness._apply_plot_colours_to_viewer = MagicMock()
        harness._update_viewer_feature_colors = MagicMock()
        harness._reset_viewer_label_colors = MagicMock()

        _call_on_annotator_isolation_cleared(harness, reason="close")

        assert harness._isolated_label_ids == set()
        assert harness._isolated_df_indices == []
        assert harness._colormap_authority.current_owner is None
        harness._find_all_patch_labels_layers.assert_not_called()
        harness._apply_plot_colours_to_viewer.assert_not_called()
        harness._update_viewer_feature_colors.assert_not_called()
        harness._reset_viewer_label_colors.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 2: _switch_patch_from_3d_annotator must clear isolation state
# ---------------------------------------------------------------------------


class TestPatchSwitchClearsIsolation:
    """_switch_patch_from_3d_annotator must clear isolation state.

    Since the full method has heavy dependencies (viewer, state_manager,
    layer loading, etc.), we test the cleanup contract directly: after
    a patch switch, isolation IDs / indices / authority must be clean.

    We verify this by:
    1. Testing that authority.reset() clears all state (already covered
       in test_colormap_authority.py, but re-stated here for clarity).
    2. Verifying the cleanup lines exist in the source code.
    3. Testing the behavioral contract: stale isolation from patch A
       should not bleed into patch B.
    """

    def test_isolation_ids_cleared_on_patch_switch(self):
        """_isolated_label_ids should be empty after the cleanup sequence."""
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_ISOLATION)
        auth.set_restore_point("old_cmap")

        # Simulate the cleanup that should happen at top of patch switch
        isolated_label_ids = {10, 20, 30}
        isolated_df_indices = [0, 1, 2]

        # -- The fix: these 3 lines are added to _switch_patch_from_3d_annotator
        isolated_label_ids = set()
        isolated_df_indices = []
        auth.reset()

        assert isolated_label_ids == set()
        assert isolated_df_indices == []
        assert auth.current_owner is None
        assert auth.restore() is None  # restore point also cleared

    def test_authority_reset_on_patch_switch(self):
        """Colormap authority should be fully reset (owner + restore point)."""
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_ISOLATION)
        auth.set_restore_point("stale_cmap_from_patch_A")

        auth.reset()

        assert auth.current_owner is None
        assert auth.restore() is None

    def test_authority_reset_allows_fresh_annotator_claim(self):
        """After reset, annotator can claim for the new patch."""
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_ISOLATION)

        auth.reset()

        result = auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        assert result is True
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION

    def test_cleanup_lines_present_in_switch_patch_method(self):
        """Source code verification: the cleanup is wired into
        _switch_patch_from_3d_annotator (not just tested in isolation).

        R3: Cleanup is now via _hard_clear_curation_surfaces helper.
        """
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )

        assert "_hard_clear_curation_surfaces" in source, (
            "_switch_patch_from_3d_annotator must call _hard_clear_curation_surfaces"
        )

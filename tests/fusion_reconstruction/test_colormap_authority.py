"""Tests for ColormapAuthority — colormap ownership protocol.

WP-0: Foundation module for coordinating colormap writes between
controller and annotator. Bugs A, C, D, I, J all depend on this.
"""

from __future__ import annotations

import pytest

from napari_epicycle.fusion_reconstruction.colormap_authority import (
    ColormapAuthority,
    ColormapOwner,
)


# ---------------------------------------------------------------------------
# WP-0 tests — ColormapAuthority basics
# ---------------------------------------------------------------------------


class TestColormapOwnerEnum:
    def test_has_all_members(self):
        assert ColormapOwner.DEFAULT == "default"
        assert ColormapOwner.CONTROLLER_QC == "controller_qc"
        assert ColormapOwner.CONTROLLER_ISOLATION == "controller_isolation"
        assert ColormapOwner.ANNOTATOR_ISOLATION == "annotator_isolation"


class TestClaimAndRelease:
    def test_claim_sets_owner(self):
        auth = ColormapAuthority()
        assert auth.current_owner is None
        result = auth.claim(ColormapOwner.CONTROLLER_QC)
        assert result is True
        assert auth.current_owner == ColormapOwner.CONTROLLER_QC

    def test_claim_by_same_owner_succeeds(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_QC)
        result = auth.claim(ColormapOwner.CONTROLLER_QC)
        assert result is True
        assert auth.current_owner == ColormapOwner.CONTROLLER_QC

    def test_claim_by_different_owner_fails_without_release(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_QC)
        result = auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        assert result is False
        assert auth.current_owner == ColormapOwner.CONTROLLER_QC

    def test_release_clears_owner(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_QC)
        auth.release(ColormapOwner.CONTROLLER_QC)
        assert auth.current_owner is None

    def test_release_by_non_owner_is_noop(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_QC)
        auth.release(ColormapOwner.ANNOTATOR_ISOLATION)
        assert auth.current_owner == ColormapOwner.CONTROLLER_QC

    def test_release_when_no_owner_is_noop(self):
        auth = ColormapAuthority()
        auth.release(ColormapOwner.CONTROLLER_QC)  # should not raise
        assert auth.current_owner is None

    def test_claim_after_release_succeeds(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_QC)
        auth.release(ColormapOwner.CONTROLLER_QC)
        result = auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        assert result is True
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION


class TestRestorePoint:
    def test_set_restore_point_and_restore(self):
        auth = ColormapAuthority()
        sentinel = object()
        auth.set_restore_point(sentinel)
        restored = auth.restore()
        assert restored is sentinel

    def test_restore_without_point_returns_none(self):
        auth = ColormapAuthority()
        assert auth.restore() is None

    def test_restore_clears_the_point(self):
        auth = ColormapAuthority()
        auth.set_restore_point("cmap_a")
        auth.restore()
        assert auth.restore() is None

    def test_set_restore_point_overwrites_previous(self):
        auth = ColormapAuthority()
        auth.set_restore_point("old")
        auth.set_restore_point("new")
        assert auth.restore() == "new"


class TestReset:
    def test_reset_clears_all_state(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        auth.set_restore_point("some_cmap")
        auth.reset()
        assert auth.current_owner is None
        assert auth.restore() is None

    def test_reset_allows_fresh_claim(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_QC)
        auth.reset()
        result = auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        assert result is True


class TestForceClaimByDefault:
    """DEFAULT owner can always be overridden without explicit release."""

    def test_default_is_initial_after_reset(self):
        auth = ColormapAuthority()
        assert auth.current_owner is None

    def test_claim_over_none_succeeds(self):
        auth = ColormapAuthority()
        assert auth.claim(ColormapOwner.CONTROLLER_ISOLATION) is True

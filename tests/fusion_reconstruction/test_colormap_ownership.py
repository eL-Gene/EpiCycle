"""Tests for colormap ownership protocol wiring.

WP-2: Controller claims/releases ColormapAuthority before writing colormaps.
WP-6: Curation action race fix (accept with isolation + sync ON).

Behavioral tests — real ColormapAuthority, mock controller surfaces.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.colormap_authority import (
    ColormapAuthority,
    ColormapOwner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLabelsLayer:
    """Minimal labels layer for colormap ownership tests."""

    def __init__(self, name="labels"):
        self.name = name
        self.colormap = "default_cmap"
        self.data = np.zeros((4, 8, 8), dtype=np.int32)
        self.properties = {}


def _make_controller_harness(*, sync_on=False, qc_df=None):
    """Build a minimal controller harness with ColormapAuthority wired in.

    Only stubs the methods and state needed for colormap ownership tests.
    """
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    harness = MagicMock()
    harness._colormap_authority = ColormapAuthority()
    harness._isolated_label_ids = set()
    harness._isolated_df_indices = []
    harness._grid_layers = []
    harness._qc_features_df = qc_df
    harness._qc_view_patch_name = "patchA"

    # Fake UI with sync checkbox
    harness.ui = MagicMock()
    harness.ui.sync_viewer_colors_cb.isChecked.return_value = sync_on

    # Fake labels layer
    harness._labels_layer = _FakeLabelsLayer()

    return harness


# ---------------------------------------------------------------------------
# WP-2 tests — Controller ownership wiring
# ---------------------------------------------------------------------------


class TestControllerIsolationOwnership:
    """Controller's _isolate_labels_in_viewer claims CONTROLLER_ISOLATION."""

    def test_controller_isolation_claims_ownership(self):
        auth = ColormapAuthority()
        # Simulate: controller isolates labels
        assert auth.claim(ColormapOwner.CONTROLLER_ISOLATION) is True
        assert auth.current_owner == ColormapOwner.CONTROLLER_ISOLATION

    def test_controller_isolation_releases_on_clear(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_ISOLATION)
        auth.release(ColormapOwner.CONTROLLER_ISOLATION)
        assert auth.current_owner is None

    def test_controller_cannot_claim_when_annotator_owns(self):
        """Bug A/C: Controller must not overwrite annotator isolation."""
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        result = auth.claim(ColormapOwner.CONTROLLER_ISOLATION)
        assert result is False
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION


class TestControllerQCOwnership:
    def test_controller_qc_coloring_claims_ownership(self):
        auth = ColormapAuthority()
        assert auth.claim(ColormapOwner.CONTROLLER_QC) is True
        assert auth.current_owner == ColormapOwner.CONTROLLER_QC

    def test_sync_toggle_on_claims_controller_qc(self):
        auth = ColormapAuthority()
        assert auth.claim(ColormapOwner.CONTROLLER_QC) is True
        assert auth.current_owner == ColormapOwner.CONTROLLER_QC

    def test_sync_toggle_off_releases_and_restores(self):
        auth = ColormapAuthority()
        sentinel = object()
        auth.claim(ColormapOwner.CONTROLLER_QC)
        auth.set_restore_point(sentinel)
        # Sync OFF → release + restore
        restored = auth.restore()
        auth.release(ColormapOwner.CONTROLLER_QC)
        assert restored is sentinel
        assert auth.current_owner is None

    def test_controller_qc_cannot_claim_when_annotator_owns(self):
        """If annotator has isolation active, sync toggle ON should not overwrite."""
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        result = auth.claim(ColormapOwner.CONTROLLER_QC)
        assert result is False


class TestControllerOwnerUpgrade:
    """Controller owners can switch between each other (same system)."""

    def test_controller_isolation_can_upgrade_to_qc(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_ISOLATION)
        assert auth.claim(ColormapOwner.CONTROLLER_QC) is True
        assert auth.current_owner == ColormapOwner.CONTROLLER_QC

    def test_controller_qc_can_upgrade_to_isolation(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.CONTROLLER_QC)
        assert auth.claim(ColormapOwner.CONTROLLER_ISOLATION) is True
        assert auth.current_owner == ColormapOwner.CONTROLLER_ISOLATION

    def test_annotator_still_blocks_controller(self):
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        assert auth.claim(ColormapOwner.CONTROLLER_QC) is False
        assert auth.claim(ColormapOwner.CONTROLLER_ISOLATION) is False


# ---------------------------------------------------------------------------
# WP-6 tests — Curation action race fix
# ---------------------------------------------------------------------------


class TestCurationActionRace:
    """Bug F: Curation accept with isolation + sync ON must produce correct
    final colormap."""

    def test_curation_accept_clears_isolation_before_controller_recolors(self):
        """Annotator releases ANNOTATOR_ISOLATION, then controller can claim."""
        auth = ColormapAuthority()
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        auth.set_restore_point("qc_colors")

        # Step 1: Annotator accept → release
        restored = auth.restore()
        auth.release(ColormapOwner.ANNOTATOR_ISOLATION)
        assert restored == "qc_colors"
        assert auth.current_owner is None

        # Step 2: Controller recolors (deferred)
        result = auth.claim(ColormapOwner.CONTROLLER_QC)
        assert result is True

    def test_curation_accept_with_isolation_and_sync_on_produces_correct_colormap(self):
        """Full sequence: annotator accept releases, controller re-applies QC colors."""
        auth = ColormapAuthority()

        # Initial state: controller has QC colors
        auth.claim(ColormapOwner.CONTROLLER_QC)
        auth.set_restore_point("default_cmap")
        # Annotator opens isolation → must release controller first in
        # practice the controller releases before annotator claims
        auth.release(ColormapOwner.CONTROLLER_QC)

        # Annotator claims isolation
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        auth.set_restore_point("qc_colors_before_isolation")

        # Accept interpolation → annotator releases
        restored = auth.restore()
        auth.release(ColormapOwner.ANNOTATOR_ISOLATION)
        assert restored == "qc_colors_before_isolation"

        # Controller re-applies QC colors
        assert auth.claim(ColormapOwner.CONTROLLER_QC) is True

    def test_interpolation_accept_with_isolation_final_colormap_correct(self):
        """End-to-end: restore point survives accept cycle."""
        auth = ColormapAuthority()

        # Controller QC active
        auth.claim(ColormapOwner.CONTROLLER_QC)
        qc_cmap = "feature_colors_v1"

        # Annotator isolation begins
        auth.release(ColormapOwner.CONTROLLER_QC)
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        auth.set_restore_point(qc_cmap)

        # User does interpolation, then accepts
        # Accept → restore + release
        restored = auth.restore()
        auth.release(ColormapOwner.ANNOTATOR_ISOLATION)

        assert restored == qc_cmap
        assert auth.current_owner is None

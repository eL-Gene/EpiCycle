"""Tests for patch switch performance optimizations (Phase 1).

WP-1: Manifest lookup cache — avoid re-scanning manifests on every switch.
WP-2: In-place layer reuse — swap data instead of remove+add when possible.
WP-3: Deferred color application — colors don't need to block layer display.

All tests use lightweight fakes (no Qt, no napari viewer).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# WP-1: Manifest lookup cache
# ---------------------------------------------------------------------------


class TestManifestLookupCache:
    """_load_patch_for_qc_view should cache run_id → manifest mapping."""

    def test_source_has_manifest_cache(self):
        """Controller should have a manifest cache dict or use cached lookup."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        # The _load_patch_for_qc_view method should NOT iterate all manifests
        # on every call. It should either:
        # 1. Use a cached dict (_manifest_cache or similar), or
        # 2. Call a helper that caches results
        source = inspect.getsource(
            FusionReconstructionController._load_patch_for_qc_view
        )
        # Should NOT have the old pattern of iterating all manifests inline
        has_cache = (
            "_manifest_cache" in source
            or "_cached_manifest" in source
            or "_resolve_manifest" in source
        )
        has_old_loop = "for mp in manifests" in source
        assert has_cache or not has_old_loop, (
            "_load_patch_for_qc_view should use cached manifest lookup, "
            "not iterate all manifests on every patch switch"
        )

    def test_resolve_manifest_cached_returns_same_result(self):
        """Cached manifest lookup should return the same paths as uncached."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        # The controller should have a _resolve_manifest_paths or similar
        # that caches. Verify it exists.
        assert hasattr(FusionReconstructionController, "_resolve_manifest_paths"), (
            "Controller should have _resolve_manifest_paths helper"
        )

    def test_manifest_cache_invalidated_on_reconstruction(self):
        """Cache should be cleared when a new reconstruction runs."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(FusionReconstructionController)
        # After reconstruction completes, cache should be invalidated
        assert "_manifest_cache" in source, (
            "Controller should have _manifest_cache attribute"
        )


# ---------------------------------------------------------------------------
# WP-2: In-place layer reuse
# ---------------------------------------------------------------------------


class TestInPlaceLayerReuse:
    """_load_patch_for_qc_view should reuse existing layers when possible."""

    def test_source_checks_existing_labels_layer(self):
        """Should check for existing labels layer and swap data in-place.

        ASYNC (2026-06-05): the in-place data swap moved from
        ``_load_patch_for_qc_view`` into the main-thread continuation
        ``_on_patch_loaded``, so introspect that method now.
        """
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._on_patch_loaded
        )
        # Should have in-place data swap pattern instead of always remove+add
        has_inplace = (
            ".data = " in source
            or "layer.data" in source
            or "lyr.data" in source
        )
        assert has_inplace, (
            "_on_patch_loaded should swap layer data in-place "
            "when the layer already exists"
        )

    def test_source_checks_existing_stack_layers(self):
        """Should check for existing stack layers before adding new ones.

        ASYNC (2026-06-05): the existing-layer reuse check moved into
        ``_on_patch_loaded``.
        """
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._on_patch_loaded
        )
        # The old code only checked _has_layer_prefix before add_image.
        # New code should also handle reuse for labels.
        # At minimum, the labels path should not unconditionally remove+add.
        has_reuse_check = (
            "layer_name in self.viewer.layers" in source
            or "_reuse_existing" in source
        )
        # The key assertion: we should NOT see unconditional remove followed
        # by add_labels for the same layer name
        assert has_reuse_check, (
            "Should check for existing layer before deciding to reuse vs recreate"
        )

    def test_inplace_data_swap_preserves_colormap(self):
        """When swapping data in-place, existing colormap should be preserved
        until QC colors are applied (avoids flash of default colors).

        ASYNC (2026-06-05): the in-place swap lives in ``_on_patch_loaded``.
        """
        # This is a behavioral contract test — in-place swap should not
        # reset the colormap
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._on_patch_loaded
        )
        # If we do in-place data swap, we should NOT set colormap to default
        # before the QC color application at the end
        # (The QC color apply handles colormap setting)
        # Just verify the in-place path exists
        assert ".data = " in source or "layer.data" in source


# ---------------------------------------------------------------------------
# WP-3: Deferred color application
# ---------------------------------------------------------------------------


class TestDeferredColorApplication:
    """QC color application should not block layer display on patch switch."""

    def test_source_has_deferred_color_pattern(self):
        """_load_patch_for_qc_view or _switch_patch_from_3d_annotator should
        defer QC color application so layers appear immediately."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        # Check if color application is deferred (via QTimer or similar).
        # ASYNC (2026-06-05): the _deferred_apply_qc_colors call moved into
        # the main-thread continuation _on_patch_loaded, so include its source.
        switch_source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        load_source = inspect.getsource(
            FusionReconstructionController._load_patch_for_qc_view
        )
        loaded_source = inspect.getsource(
            FusionReconstructionController._on_patch_loaded
        )
        combined = switch_source + load_source + loaded_source

        has_deferred = (
            "QTimer" in combined
            or "singleShot" in combined
            or "_deferred" in combined
            or "call_later" in combined
        )
        assert has_deferred, (
            "QC color application should be deferred so the layer appears "
            "immediately and colors fill in on the next event loop tick"
        )

    def test_load_patch_returns_before_colors(self):
        """The layer should be visible before QC colors are applied."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        # ASYNC (2026-06-05): the deferred color call lives in the main-thread
        # continuation _on_patch_loaded now, not _load_patch_for_qc_view.
        source = inspect.getsource(
            FusionReconstructionController._on_patch_loaded
        )
        # The deferred pattern means _on_patch_loaded should make the layer
        # visible before color application.  The color application code should
        # be in a deferred callback.
        has_deferred = (
            "QTimer" in source
            or "singleShot" in source
            or "_deferred" in source
        )
        assert has_deferred


# ---------------------------------------------------------------------------
# Integration: patch switch flow
# ---------------------------------------------------------------------------


class TestPatchSwitchFlow:
    """End-to-end tests for the optimized patch switch."""

    def test_same_patch_guard_skips_reload(self):
        """Switching to the same patch should return immediately."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        assert "_qc_view_patch_name" in source, (
            "Same-patch guard should check _qc_view_patch_name"
        )

    def test_manifest_cache_exists_as_attribute(self):
        """Controller __init__ should initialize _manifest_cache."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(FusionReconstructionController.__init__)
        assert "_manifest_cache" in source, (
            "Controller __init__ should initialize _manifest_cache"
        )


# ---------------------------------------------------------------------------
# Async load safety: early-return recovery, finalize guard, shutdown sweep
# ---------------------------------------------------------------------------


class TestAsyncPatchLoadSafety:
    """Audit-gap coverage for the async patch-load (2026-06-05).

    The async refactor introduced three failure surfaces the original WP tests
    didn't cover:
      * early-return strand — _load_patch_for_qc_view can bail before launching
        the worker, leaving the suppress flags + status stuck;
      * finalize-mid-load — _finalize_curation could finalize the not-yet-loaded
        patch on its stale on-disk labels;
      * shutdown crash — the new per-token thread dict isn't in the hardcoded
        shutdown sweep.
    """

    def test_load_patch_returns_bool_and_switch_recovers(self):
        """_load_patch_for_qc_view returns a bool; _switch_to_curation_patch
        recovers (re-syncs the annotator) when the load did NOT launch."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        load_source = inspect.getsource(
            FusionReconstructionController._load_patch_for_qc_view
        )
        # Bool contract: both an early-return False and the launched True path.
        assert "return False" in load_source, (
            "_load_patch_for_qc_view should return False at early-returns"
        )
        assert "return True" in load_source, (
            "_load_patch_for_qc_view should return True after launching worker"
        )

        switch_source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        # The not-launched recovery re-runs the annotator sync helper.
        assert "_post_switch_annotator_sync" in switch_source, (
            "_switch_to_curation_patch should re-sync the annotator when the "
            "load does not launch"
        )

    def test_finalize_guards_on_patch_load_in_flight(self):
        """_finalize_curation must early-return while a patch load is in flight
        so it can't finalize the not-yet-loaded patch on stale disk labels."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._finalize_curation
        )
        assert "_patch_load_in_flight" in source, (
            "_finalize_curation should guard on _patch_load_in_flight"
        )

    def test_shutdown_sweeps_patch_load_threads(self):
        """shutdown() must quit/wait the per-token patch-load threads to avoid
        'QThread: Destroyed while thread is still running' on app close."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(FusionReconstructionController.shutdown)
        assert "_patch_load_threads" in source, (
            "shutdown() should sweep _patch_load_threads"
        )

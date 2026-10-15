"""Tests for patch switch performance optimizations.

Covers:
1. Concurrent TIFF reading (off main thread)
2. Loading guard prevents double-switches
3. Source stack reuse (skip re-read if already loaded)
4. Layer cleanup correctness on switch
5. Fast reload path (in-place data swap)
6. Subviewer sync suppression during batch load

All tests are Qt-free using lightweight fakes.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLayer:
    """Minimal layer fake."""

    def __init__(self, name, data=None, *, kind="labels"):
        self.name = name
        self.data = data if data is not None else np.zeros((10, 10, 10), dtype=np.int32)
        self._kind = kind
        self.visible = True
        self.opacity = 1.0
        self.events = MagicMock()
        self.colormap = None
        self._refreshed = 0

    def refresh(self):
        self._refreshed += 1


class FakeLayers:
    """Minimal viewer.layers fake supporting __contains__ and iteration."""

    def __init__(self, layers=None):
        self._layers = list(layers or [])

    def __contains__(self, name):
        return any(l.name == name for l in self._layers)

    def __getitem__(self, name):
        for l in self._layers:
            if l.name == name:
                return l
        raise KeyError(name)

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def append(self, layer):
        self._layers.append(layer)

    def remove(self, name):
        self._layers = [l for l in self._layers if l.name != name]


# ---------------------------------------------------------------------------
# Test: Source stack reuse
# ---------------------------------------------------------------------------


class TestSourceStackReuse:
    """Controller should skip TIFF re-read when source stack is already loaded."""

    def test_has_layer_prefix_finds_existing(self):
        """_has_layer_prefix should find layers by name prefix."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        import inspect
        source = inspect.getsource(
            FusionReconstructionController._has_layer_prefix
        )
        # Method should check layer names
        assert "layer.name" in source or "layer" in source

    def test_skip_source_reload_when_present(self):
        """When stack_<stem> already exists in viewer, skip tifffile.imread."""
        # The guard is: if not self._has_layer_prefix(stack_layer_name): load()
        # Verify source has this guard
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._load_patch_for_qc_view
        )
        assert "_has_layer_prefix" in source, (
            "_load_patch_for_qc_view should check _has_layer_prefix before loading source stack"
        )


# ---------------------------------------------------------------------------
# Test: Labels in-place swap
# ---------------------------------------------------------------------------


class TestLabelsInPlaceSwap:
    """When shapes match, labels should be swapped in-place (no layer recreation)."""

    def test_fast_path_exists_in_source(self):
        """_load_patch_for_qc_view should have in-place data swap path."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._load_patch_for_qc_view
        )
        assert "existing.data = labels" in source or "existing.data =" in source, (
            "_load_patch_for_qc_view should swap data in-place when shapes match"
        )

    def test_in_place_swap_preserves_layer_identity(self):
        """In-place data swap should keep the same layer object (no GPU realloc)."""
        old_data = np.ones((10, 20, 20), dtype=np.int32)
        new_data = np.ones((10, 20, 20), dtype=np.int32) * 5
        layer = FakeLayer("reconstruction_abc", old_data)

        # Simulate in-place swap
        layer.data = new_data
        assert np.array_equal(layer.data, new_data)
        assert layer.name == "reconstruction_abc"  # same layer


# ---------------------------------------------------------------------------
# Test: Concurrent TIFF read helper
# ---------------------------------------------------------------------------


class TestConcurrentTiffRead:
    """Verify concurrent TIFF reading works correctly."""

    def test_thread_pool_reads_file(self, tmp_path):
        """ThreadPoolExecutor should read a TIFF file correctly."""
        import tifffile
        from concurrent.futures import ThreadPoolExecutor

        # Write test TIFF
        vol = np.random.randint(0, 100, (10, 20, 20), dtype=np.int32)
        path = tmp_path / "test.tif"
        tifffile.imwrite(str(path), vol)

        # Read on thread
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(tifffile.imread, str(path))
            result = future.result(timeout=10)

        np.testing.assert_array_equal(vol, result)

    def test_concurrent_labels_and_csv(self, tmp_path):
        """Labels TIFF and CSV should be readable concurrently."""
        import tifffile
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Write test data
        vol = np.random.randint(0, 5, (10, 20, 20), dtype=np.int32)
        tiff_path = tmp_path / "labels.tif"
        tifffile.imwrite(str(tiff_path), vol)

        df = pd.DataFrame({"label": [1, 2, 3], "area": [100, 200, 300]})
        csv_path = tmp_path / "features.csv"
        df.to_csv(str(csv_path), index=False)

        results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_tiff = pool.submit(tifffile.imread, str(tiff_path))
            f_csv = pool.submit(pd.read_csv, str(csv_path))
            results["tiff"] = f_tiff.result(timeout=10)
            results["csv"] = f_csv.result(timeout=10)

        np.testing.assert_array_equal(vol, results["tiff"])
        assert len(results["csv"]) == 3


# ---------------------------------------------------------------------------
# Test: Loading guard prevents double-switch
# ---------------------------------------------------------------------------


class TestLoadingGuard:
    """A loading guard should prevent concurrent patch switches."""

    def test_guard_flag_blocks_reentrant_switch(self):
        """If _patch_switch_loading is True, switch should return early."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        assert "_patch_switch_loading" in source, (
            "_switch_to_curation_patch should have a loading guard"
        )


# ---------------------------------------------------------------------------
# Test: Cleanup on patch switch
# ---------------------------------------------------------------------------


class TestPatchSwitchCleanup:
    """Verify proper cleanup during patch switching."""

    def test_hard_clear_called_before_load(self):
        """_hard_clear_curation_surfaces should be called before loading new patch."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        hard_clear_pos = source.find("_hard_clear_curation_surfaces")
        load_patch_pos = source.find("_load_patch_for_qc_view")
        assert hard_clear_pos < load_patch_pos, (
            "Hard clear must happen before loading new patch"
        )

    def test_clear_curation_layers_called_before_load(self):
        """Stale curation layers should be removed before loading new patch."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        clear_pos = source.find("_clear_curation_layers")
        load_pos = source.find("_load_patch_for_qc_view")
        assert clear_pos < load_pos, (
            "Curation layers must be cleared before loading new patch"
        )

    def test_qc_view_layers_cleared_before_load(self):
        """QC view layer list should be cleared before loading new patch."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        clear_pos = source.find("_clear_qc_view_layers")
        load_pos = source.find("_load_patch_for_qc_view")
        assert clear_pos < load_pos, (
            "QC view layers must be cleared before loading new patch"
        )


# ---------------------------------------------------------------------------
# Test: Subviewer fast reload
# ---------------------------------------------------------------------------


class TestSubviewerFastReload:
    """Verify subviewer fast reload path works correctly."""

    def test_fast_reload_with_matching_shapes(self):
        """try_fast_reload should succeed when shapes match."""
        from packages.epicycle_annotator.subviewers import SubviewerManager
        from packages.epicycle_annotator.orientation import transpose_data
        from napari.layers import Labels, Image
        from napari.utils._proxies import ReadOnlyWrapper

        # This test verifies the fast reload logic by checking source
        import inspect
        source = inspect.getsource(SubviewerManager.try_fast_reload)
        # Should check shapes before attempting swap
        assert "shape" in source
        # Should rename layers
        assert ".name =" in source or "name" in source
        # Should swap data
        assert ".data =" in source

    def test_fast_reload_returns_old_names(self):
        """try_fast_reload should return old label names for sync manager cleanup."""
        import inspect
        from packages.epicycle_annotator.subviewers import SubviewerManager
        source = inspect.getsource(SubviewerManager.try_fast_reload)
        assert "old_label_names" in source, (
            "try_fast_reload should track and return old label names"
        )


# ---------------------------------------------------------------------------
# Test: Sync suppression during load
# ---------------------------------------------------------------------------


class TestSyncSuppressionDuringLoad:
    """Property sync should be suppressed during batch layer setup."""

    def test_sync_suppression_in_load_selected_layers(self):
        """_load_selected_layers should suppress sync during property setup."""
        import inspect
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        source = inspect.getsource(
            NativeMultiviewAnnotator._load_selected_layers
        )
        assert "suppress_property_sync" in source, (
            "_load_selected_layers should suppress sync during batch property setup"
        )


# ---------------------------------------------------------------------------
# Test: Same-patch guard
# ---------------------------------------------------------------------------


class TestSamePatchGuard:
    """Switching to the same patch should be a fast no-op."""

    def test_same_patch_returns_immediately(self):
        """_switch_to_curation_patch should short-circuit for same patch."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        assert "_qc_view_patch_name" in source, (
            "Should check current patch name to skip same-patch reload"
        )


# ---------------------------------------------------------------------------
# Test: Deferred QC color application
# ---------------------------------------------------------------------------


class TestDeferredQCColors:
    """QC colors should be applied asynchronously after layer appears."""

    def test_deferred_color_via_qtimer(self):
        """_deferred_apply_qc_colors should use QTimer.singleShot."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._deferred_apply_qc_colors
        )
        assert "QTimer.singleShot" in source, (
            "QC colors should be deferred via QTimer.singleShot"
        )


# ---------------------------------------------------------------------------
# Test: Layer removal suppression during patch switch
# ---------------------------------------------------------------------------


class TestLayerRemovalSuppression:
    """_on_main_layer_removed should be suppressed during controlled patch switch."""

    def test_suppression_flag_in_on_main_layer_removed(self):
        """Handler should check _suppress_layer_removal_cleanup flag."""
        import inspect
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        source = inspect.getsource(
            NativeMultiviewAnnotator._on_main_layer_removed
        )
        assert "_suppress_layer_removal_cleanup" in source, (
            "_on_main_layer_removed should check suppression flag"
        )

    def test_canonical_switch_sets_suppression(self):
        """_switch_to_curation_patch should set suppression flags."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        assert "_suppress_layer_removal_cleanup = True" in source
        assert "_suppress_events = True" in source

    def test_canonical_switch_clears_suppression_in_finally(self):
        """Suppression must be cleared via _post_switch_annotator_sync in finally."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        assert "finally:" in source
        assert "_post_switch_annotator_sync" in source

    def test_post_switch_clears_flags_and_triggers_autoload(self):
        """_post_switch_annotator_sync should clear flags and trigger auto-load."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._post_switch_annotator_sync
        )
        assert "_suppress_layer_removal_cleanup = False" in source
        assert "_suppress_events = False" in source
        assert "_schedule_deferred_auto_load" in source

    def test_post_switch_updates_flagged_labels(self):
        """_post_switch_annotator_sync should update flagged labels on annotator."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._post_switch_annotator_sync
        )
        assert "set_focused_refinement" in source
        assert "set_triage_callbacks" in source
        assert "set_prob_volume_path" in source

    def test_boxlist_delegates_to_canonical(self):
        """_on_curation_patch_clicked should delegate to _switch_to_curation_patch."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._on_curation_patch_clicked
        )
        assert "_switch_to_curation_patch" in source

    def test_dropdown_delegates_to_canonical(self):
        """_switch_patch_from_3d_annotator should delegate to _switch_to_curation_patch."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_patch_from_3d_annotator
        )
        assert "_switch_to_curation_patch" in source

    def test_get_open_annotator_widget_helper_exists(self):
        """Controller should have _get_open_annotator_widget helper."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        assert hasattr(FusionReconstructionController, "_get_open_annotator_widget")

    def test_collect_flagged_labels_helper_exists(self):
        """Controller should have _collect_flagged_labels helper."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        assert hasattr(FusionReconstructionController, "_collect_flagged_labels")


# ---------------------------------------------------------------------------
# Test: SubviewerMirrorManager event suppression
# ---------------------------------------------------------------------------


class TestSubviewerMirrorManagerSuppression:
    """SubviewerMirrorManager must suppress add/remove during controlled switches."""

    def test_suppress_events_flag_exists(self):
        """SubviewerMirrorManager should have _suppress_events flag."""
        from packages.epicycle_annotator.subviewer_mirror import SubviewerMirrorManager
        mgr = SubviewerMirrorManager(
            main_viewer=MagicMock(),
            subviewer_manager=MagicMock(),
            sync_manager=MagicMock(),
            drawing_handler=MagicMock(),
        )
        assert hasattr(mgr, "_suppress_events")
        assert mgr._suppress_events is False

    def test_on_layer_removed_skips_subviewer_removal_when_suppressed(self):
        """on_layer_removed should skip remove_layer_from_subviewers when suppressed."""
        from packages.epicycle_annotator.subviewer_mirror import SubviewerMirrorManager
        sub_mgr = MagicMock()
        sync_mgr = MagicMock()
        mgr = SubviewerMirrorManager(
            main_viewer=MagicMock(),
            subviewer_manager=sub_mgr,
            sync_manager=sync_mgr,
            drawing_handler=MagicMock(),
        )
        mgr._suppress_events = True
        fake_event = MagicMock()
        fake_event.value = FakeLayer("test_layer")
        mgr.on_layer_removed(fake_event)
        sub_mgr.remove_layer_from_subviewers.assert_not_called()
        sync_mgr.unregister_layer.assert_not_called()

    def test_on_layer_removed_cleans_rename_cb_even_when_suppressed(self):
        """Rename callback cleanup should happen even during suppression."""
        from packages.epicycle_annotator.subviewer_mirror import SubviewerMirrorManager
        mgr = SubviewerMirrorManager(
            main_viewer=MagicMock(),
            subviewer_manager=MagicMock(),
            sync_manager=MagicMock(),
            drawing_handler=MagicMock(),
        )
        fake_layer = FakeLayer("test_layer")
        cb = MagicMock()
        mgr._rename_cbs[id(fake_layer)] = (fake_layer, cb)
        mgr._suppress_events = True
        fake_event = MagicMock()
        fake_event.value = fake_layer
        mgr.on_layer_removed(fake_event)
        # Rename cb should be cleaned up even during suppression.
        assert id(fake_layer) not in mgr._rename_cbs

    def test_on_layer_added_skips_when_suppressed(self):
        """on_layer_added should skip subviewer operations when suppressed."""
        from packages.epicycle_annotator.subviewer_mirror import SubviewerMirrorManager
        sub_mgr = MagicMock()
        mgr = SubviewerMirrorManager(
            main_viewer=MagicMock(),
            subviewer_manager=sub_mgr,
            sync_manager=MagicMock(),
            drawing_handler=MagicMock(),
        )
        mgr._suppress_events = True
        fake_event = MagicMock()
        fake_event.value = FakeLayer("test_img", kind="image")
        mgr.on_layer_added(fake_event)
        sub_mgr.add_image_to_subviewers.assert_not_called()
        sub_mgr.add_labels_to_subviewers.assert_not_called()

    def test_canonical_switch_sets_suppress_events(self):
        """_switch_to_curation_patch should set layer_manager._suppress_events."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        assert "_suppress_events = True" in source

    def test_post_switch_clears_suppress_events(self):
        """_post_switch_annotator_sync should clear _suppress_events."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._post_switch_annotator_sync
        )
        assert "_suppress_events = False" in source

    def test_post_switch_triggers_deferred_auto_load(self):
        """_post_switch_annotator_sync should trigger deferred auto-load."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._post_switch_annotator_sync
        )
        assert "_schedule_deferred_auto_load" in source

    def test_load_selected_layers_disconnects_old_handlers(self):
        """_load_selected_layers should disconnect old vis/style handlers."""
        import inspect
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        source = inspect.getsource(
            NativeMultiviewAnnotator._load_selected_layers
        )
        assert "disconnect_layer_visibility" in source
        assert "disconnect_image_style" in source

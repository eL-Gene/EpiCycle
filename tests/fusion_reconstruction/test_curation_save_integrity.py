"""Tests for curation save/load integrity — mask persistence, atomic writes,
discard revert, stale layer cleanup, and CSV/volume reconciliation.

These test real controller methods bound to a minimal harness, exercising
the actual production save/load code paths.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal controller harness — reuses production methods
# ---------------------------------------------------------------------------

class _ControllerHarness:
    """Wraps actual controller methods with minimal fake state."""

    def __init__(self, tmp_path):
        self.working_dir = str(tmp_path)
        self._curation_dirty = False
        self._tiers_dirty = False
        self._edited_patches = set()
        self._qc_features_df = None
        self._qc_view_patch_name = ""
        self._curation_undo_stack = MagicMock()
        self._curation_undo_stack.__bool__ = MagicMock(return_value=False)
        self._curation_undo_stack.clear = MagicMock()
        self._committing_triage = False
        # A1: the thumbnail re-trigger after a labels write (not under test here)
        self._refresh_step3_thumbs_after_labels = lambda why: None
        self._tier_undo_stack = MagicMock()
        self._tier_undo_stack.clear = MagicMock()
        # 2026-06-07: commit-triage async flags (gate re-entry + stale-finish
        # token).  The synchronous _commit_triage_single_patch path leaves the
        # in-flight flag False; only the async _commit_triage button path sets
        # it.  Token starts at 0 so a sync finish (token=current) is never stale.
        self._commit_triage_in_flight = False
        self._commit_triage_token = 0
        self._commit_triage_dispatch_patch = ""
        self._commit_triage_thread = None
        self._commit_triage_worker = None
        # 2026-06-07: save-curation async flags (reuses _save_in_flight as the
        # in-flight gate + stale-finish token).  The synchronous _save_curation
        # path sets _save_in_flight only briefly during its inline write; only
        # the async _save_curation_async button path holds it across a QThread.
        # Token starts at 0 so a sync finish (token=current) is never stale.
        self._save_in_flight = False
        self._save_dispatch_patch = None
        self._save_curation_token = 0
        self._save_curation_dispatch_patch = ""
        self._save_curation_thread = None
        self._save_curation_worker = None
        self._log = []

        # Fake UI
        self.ui = MagicMock()
        self.ui.save_curation_btn = MagicMock()
        self.ui.undo_tier_btn = MagicMock()
        # qc_plot_widget mock needs real .count() returns for combo guards
        pw = self.ui.qc_plot_widget
        pw._color_combo.count.return_value = 0
        pw._cmap_combo.count.return_value = 0

        # Fake viewer with layers
        self.viewer = MagicMock()
        self._labels_layer = None
        self._labels_path = None

        # Fake state manager
        self.state_manager = MagicMock()
        self.state_manager._state = {"patches": {}}
        self.state_manager.get_patch_status = MagicMock(return_value="curating")
        # D2 (perf push 2026-05-09): default to no-downsample so existing
        # save tests behave as before.  Tests that exercise the lift
        # path explicitly override these.
        self.state_manager.get_patch_downsample_factor = MagicMock(return_value=1)
        self.state_manager.get_patch_original_labels_shape = MagicMock(
            return_value=None
        )

        # Fake run store
        self.run_store = MagicMock()
        self.run_store.list_manifests = MagicMock(return_value=[])

        # Colormap authority (needed by _switch_patch_from_3d_annotator)
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
        )
        self._colormap_authority = ColormapAuthority()
        self._isolated_label_ids = set()
        self._isolated_df_indices = []

        # QC view tracking
        self._qc_view_layers = []
        self._grid_layers = []

    def _append_log(self, msg):
        self._log.append(msg)

    def _get_navigator_widget(self):
        return None

    def _get_open_annotator_widget(self):
        return None

    def _hard_clear_curation_surfaces(self):
        """Stub for the unified hard-clear helper."""
        self._isolated_label_ids = set()
        self._isolated_df_indices = []
        self._colormap_authority.reset()

    def _find_active_labels_layer(self):
        return self._labels_layer

    def _get_active_labels_path(self):
        return self._labels_path

    def _selected_curation_patch(self):
        return getattr(self, "_cur_patch", None)

    def _selected_patch_name(self):
        return getattr(self, "_patch_name", None)

    # _is_population_mode harness stub removed 2026-05-03
    # (drop-population-mode Commit 5/6) — production method gone.

    def _refresh_patch_list_colors(self):
        pass

    def _refresh_summary_dashboard(self):
        pass

    def _refresh_patch_state_list(self):
        pass

    def _refresh_curation_ui(self):
        pass

    def _auto_assign_qc_tiers(self, df):
        pass

    def _update_qc_summary(self, df):
        pass

    def _populate_threshold_feature_combo(self, df):
        pass

    def _update_flagged_labels_list(self):
        pass

    def _update_label_basic_stats(self, label_id, *, roi_slices=None):
        pass

    def _select_curation_bucket_item(self, name):
        pass

    def _clear_qc_view_layers(self):
        self._qc_view_layers.clear()
        self._qc_view_patch_name = ""

    def _clear_curation_layers(self):
        pass

    def _load_patch_for_qc_view(self, patch_name=""):
        pass

    def _has_layer_prefix(self, prefix):
        return False


def _bind_real_methods(harness, method_names):
    """Import and bind real controller methods to the harness."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    import types

    for method_name in method_names:
        real_method = getattr(FusionReconstructionController, method_name)
        harness.__dict__[method_name] = types.MethodType(real_method, harness)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path):
    h = _ControllerHarness(tmp_path)
    _bind_real_methods(h, [
        "_save_curation",
        "_save_curation_checked",
        "_save_tiers_csv",
        # 2026-06-07: _save_curation went async (Save BUTTON → background
        # QThread persist).  The SYNCHRONOUS _save_curation is now a thin
        # orchestrator (prep → pure persist → finalize) that the disk /
        # transactional tests drive without a live QThread.  Bind the prep +
        # finalize/error helpers it delegates to, plus the async launcher +
        # click handler the new in-flight-gate + stale-token tests call.
        "_save_curation_prepare",
        "_save_curation_async",
        "_on_save_curation_clicked",
        "_on_save_curation_finished",
        "_on_save_curation_error",
        "_commit_triage_single_patch",
        # 2026-06-07: commit-triage went async.  _commit_triage_single_patch
        # is now a thin synchronous orchestrator (prep → pure persist →
        # finalize) that the disk/transactional tests drive without a live
        # QThread.  Bind the prep + finalize helpers it delegates to, plus the
        # async finish/error handlers the new in-flight-gate + stale-token
        # tests call directly.
        "_commit_triage",
        "_commit_triage_prepare",
        "_on_commit_triage_finished",
        "_on_commit_triage_error",
        "_commit_triage_partial_failure_refresh",
        "_revert_labels_from_disk",
        "_reload_current_patch_features",
        "_check_csv_volume_consistency",
        "_maybe_lift_for_save",  # D2 helper used by all 3 save sites
        # 2026-05-17 fix: save sites call this to refresh the save
        # button enabled state based on both dirty flags.
        "_refresh_save_button_state",
        # 2026-05-17 diagnostic: save sites + load sites scan for
        # duplicate label rows.  Read-only.
        "_scan_qc_df_for_duplicates",
        # 2026-06-05 (integration/step3-patch-perf): the generalized busy
        # bar.  _save_curation_checked + _switch_to_curation_patch now call
        # these; bind them so harnesses exercising those paths don't
        # AttributeError.  Self-contained (getattr/try-guarded on self.ui).
        "_busy_show",
        "_busy_progress",
        "_busy_hide",
    ])
    # The CSV/volume consistency check is perf-gated OFF by default in
    # production (full-volume np.unique on every reload).  Opt the test
    # harness in so the mismatch-warning behavior tests actually exercise it.
    h._DEBUG_CSV_VOLUME_CONSISTENCY = True
    return h


# ---------------------------------------------------------------------------
# D2 — Save-time lift behavior
# ---------------------------------------------------------------------------


class TestMaybeLiftForSave:
    """The save-lift helper consults state_manager and NN-upsamples
    the in-memory mask back to original full-res shape when this
    patch was loaded with downsample_factor > 1.  Used by all three
    save sites (staging, pre-commit snapshot, commit-triage)."""

    def test_factor_1_returns_input_unchanged(self, harness):
        import numpy as np
        data = np.array([[[1, 2], [3, 4]]], dtype=np.uint32)
        # Default mock: factor=1 → no lift.
        out = harness._maybe_lift_for_save(data, "patch_a")
        # Same object identity — no copy when no lift needed.
        assert out is data

    def test_factor_2_lifts_to_target_shape(self, harness):
        import numpy as np
        # Downsampled volume — shape (1, 2, 2)
        data = np.array([[[5, 7], [9, 3]]], dtype=np.uint32)
        # Pretend this patch was loaded with factor=2 from a (2, 4, 4)
        # original.
        harness.state_manager.get_patch_downsample_factor = MagicMock(
            return_value=2
        )
        harness.state_manager.get_patch_original_labels_shape = MagicMock(
            return_value=(2, 4, 4)
        )
        out = harness._maybe_lift_for_save(data, "patch_a")
        # NN-upsample produces (2, 4, 4) — each input voxel → 2³ block.
        assert out.shape == (2, 4, 4)
        # Top-left 2×2×2 should be all 5 (the (0,0,0) input voxel).
        assert (out[0:2, 0:2, 0:2] == 5).all()
        assert (out[0:2, 0:2, 2:4] == 7).all()

    def test_factor_2_with_no_target_shape_falls_back(self, harness):
        """If state_manager returns None for original shape (defensive
        fallback), helper uses factor-multiplied current shape."""
        import numpy as np
        data = np.array([[[1, 2], [3, 4]]], dtype=np.uint32)
        harness.state_manager.get_patch_downsample_factor = MagicMock(
            return_value=2
        )
        harness.state_manager.get_patch_original_labels_shape = MagicMock(
            return_value=None
        )
        out = harness._maybe_lift_for_save(data, "patch_a")
        # Fallback: shape * factor → (2, 4, 4)
        assert out.shape == (2, 4, 4)

    def test_factor_2_dtype_preserved(self, harness):
        import numpy as np
        data = np.zeros((1, 2, 2), dtype=np.int32)
        harness.state_manager.get_patch_downsample_factor = MagicMock(
            return_value=2
        )
        harness.state_manager.get_patch_original_labels_shape = MagicMock(
            return_value=(2, 4, 4)
        )
        out = harness._maybe_lift_for_save(data, "patch_a")
        assert out.dtype == np.int32


class TestSaveCurationWithDownsample:
    """End-to-end: _save_curation called on a downsampled patch lifts
    the mask to full-res before writing the canonical TIFF."""

    def test_downsampled_save_produces_full_res_tiff(self, harness, tmp_path):
        import numpy as np
        import tifffile

        # Set up a downsampled patch state.
        labels_dir = tmp_path / "artifacts" / "step3" / "runs" / "run001"
        labels_dir.mkdir(parents=True)
        labels_path = labels_dir / "labels_3d.tif"
        # Pre-existing full-res mask on disk (will be overwritten).
        tifffile.imwrite(
            str(labels_path),
            np.zeros((4, 8, 8), dtype=np.uint32),
        )

        # Set up an in-memory downsampled labels layer (shape 2,4,4).
        ds_data = np.full((2, 4, 4), 7, dtype=np.uint32)
        layer = MagicMock()
        layer.data = ds_data
        layer.refresh = MagicMock()
        harness._labels_layer = layer
        harness._labels_path = str(labels_path)
        harness._qc_view_patch_name = "patch_a"
        harness._curation_dirty = True
        harness.state_manager.get_patch_downsample_factor = MagicMock(
            return_value=2
        )
        harness.state_manager.get_patch_original_labels_shape = MagicMock(
            return_value=(4, 8, 8)
        )
        # Mock CSV side effects: no df → save skips CSV write (mask only).
        harness._qc_features_df = None

        harness._save_curation()

        # On-disk TIFF must be FULL-RES (4, 8, 8), not downsampled (2, 4, 4)
        loaded = tifffile.imread(str(labels_path))
        assert loaded.shape == (4, 8, 8)
        # And every voxel should be 7 (NN-upsample from uniform-7 input).
        assert (loaded == 7).all()


def _setup_labels(harness, tmp_path, label_data=None):
    """Set up a fake labels layer + path so mask writes succeed."""
    if label_data is None:
        label_data = np.zeros((4, 8, 8), dtype=np.uint32)
        label_data[1, 2:5, 2:5] = 1
        label_data[2, 3:6, 3:6] = 2
    layer = MagicMock()
    layer.data = label_data
    layer.refresh = MagicMock()
    harness._labels_layer = layer
    labels_dir = tmp_path / "artifacts" / "step3" / "runs" / "run001"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / "labels_3d.tif"
    harness._labels_path = str(labels_path)
    return layer, labels_path


def _write_initial_labels(labels_path, data):
    """Write initial labels TIFF to disk (simulates prior save/reconstruction)."""
    import tifffile
    tifffile.imwrite(str(labels_path), data.astype(np.uint32), bigtiff=True)


def _setup_csv(harness, tmp_path, patch_name="test_patch.tif", labels=None):
    """Set up a QC DataFrame + patch name so CSV writes succeed."""
    if labels is None:
        labels = [1, 2]
    harness._qc_features_df = pd.DataFrame({
        "label": labels,
        "area": [100] * len(labels),
        "qc_tier": ["good"] * len(labels),
        "feature_state": ["fresh"] * len(labels),
    })
    harness._cur_patch = patch_name
    harness._patch_name = patch_name
    (tmp_path / "artifacts" / "step3" / "qc").mkdir(parents=True, exist_ok=True)


# ===========================================================================
# #2: Commit triage ALWAYS saves mask, not just when garbage exists
# ===========================================================================

class TestCommitTriageAlwaysSavesMask:
    """_commit_triage_single_patch must persist the label volume regardless
    of whether any garbage labels were discarded."""

    def test_no_garbage_still_writes_mask(self, harness, tmp_path):
        """Commit with zero garbage labels must still write the mask TIFF."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        harness._tiers_dirty = True

        harness._commit_triage_single_patch()

        assert labels_path.exists(), (
            "Mask TIFF must be written even when no garbage labels exist"
        )
        assert harness._curation_dirty is False
        assert harness._tiers_dirty is False

    def test_no_garbage_mask_matches_layer_data(self, harness, tmp_path):
        """Written mask must match the in-memory layer data exactly."""
        import tifffile

        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        data[2, 3:6, 3:6] = 2
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data)
        _setup_csv(harness, tmp_path, labels=[1, 2])

        harness._commit_triage_single_patch()

        written = tifffile.imread(str(labels_path))
        np.testing.assert_array_equal(written, data)

    def test_with_garbage_still_writes_mask(self, harness, tmp_path):
        """Commit with garbage labels writes mask with garbage zeroed."""
        import tifffile

        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        data[2, 3:6, 3:6] = 2
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._qc_features_df["qc_tier"] = ["good", "garbage"]

        # Need _zero_labels_in_viewer for garbage path
        def _zero(ids):
            for lid in ids:
                layer.data[layer.data == lid] = 0
        harness._zero_labels_in_viewer = _zero

        harness._commit_triage_single_patch()

        assert labels_path.exists()
        written = tifffile.imread(str(labels_path))
        assert 2 not in written, "Garbage label 2 should be zeroed"
        assert 1 in written, "Good label 1 should survive"

    def test_csv_updated_after_commit(self, harness, tmp_path):
        """CSV must be written and garbage rows marked as deleted (ghost)."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2, 3])
        harness._qc_features_df["qc_tier"] = ["good", "garbage", "review"]

        def _zero(ids):
            for lid in ids:
                harness._labels_layer.data[harness._labels_layer.data == lid] = 0
        harness._zero_labels_in_viewer = _zero

        harness._commit_triage_single_patch()

        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "test_patch_features.csv"
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        # All 3 rows should be present (garbage kept as ghost)
        assert set(loaded["label"].values) == {1, 2, 3}
        # Garbage label should be marked deleted
        garbage_row = loaded[loaded["label"] == 2]
        assert garbage_row.iloc[0]["label_status"] == "deleted"
        # Others should be active
        active = loaded[loaded["label_status"] == "active"]
        assert set(active["label"].values) == {1, 3}


# ===========================================================================
# #3: Atomic TIFF writes — crash safety
# ===========================================================================

class TestAtomicTiffWrite:
    """_atomic_tiff_write must not leave corrupted files on failure."""

    def test_successful_write(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        import tifffile

        path = tmp_path / "test.tif"
        data = np.ones((4, 8, 8), dtype=np.uint32)

        _atomic_tiff_write(path, data)

        assert path.exists()
        loaded = tifffile.imread(str(path))
        np.testing.assert_array_equal(loaded, data)

    def test_overwrites_existing_atomically(self, tmp_path):
        """Overwriting an existing file must succeed without corruption."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        import tifffile

        path = tmp_path / "test.tif"
        old_data = np.zeros((4, 8, 8), dtype=np.uint32)
        _atomic_tiff_write(path, old_data)

        new_data = np.ones((4, 8, 8), dtype=np.uint32) * 42
        _atomic_tiff_write(path, new_data)

        loaded = tifffile.imread(str(path))
        np.testing.assert_array_equal(loaded, new_data)

    def test_failure_preserves_original(self, tmp_path):
        """If the write fails, the original file must remain intact."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        import tifffile

        path = tmp_path / "test.tif"
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        _atomic_tiff_write(path, original)

        # Force failure by patching tifffile.imwrite to raise
        with patch("tifffile.imwrite", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                _atomic_tiff_write(path, np.ones((4, 8, 8), dtype=np.uint32))

        # Original must still be intact
        loaded = tifffile.imread(str(path))
        np.testing.assert_array_equal(loaded, original)

    def test_no_temp_file_left_on_failure(self, tmp_path):
        """Failed write must not leave temporary files behind."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )

        path = tmp_path / "test.tif"
        with patch("tifffile.imwrite", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                _atomic_tiff_write(path, np.ones((4, 8, 8), dtype=np.uint32))

        # Only allow the target file (if it existed) — no .tif temp files
        tif_files = list(tmp_path.glob("*.tif"))
        assert len(tif_files) == 0, f"Temp files left behind: {tif_files}"

    def test_creates_parent_dirs(self, tmp_path):
        """Must create parent directories if they don't exist."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )

        path = tmp_path / "deep" / "nested" / "test.tif"
        data = np.ones((2, 4, 4), dtype=np.uint32)

        _atomic_tiff_write(path, data)

        assert path.exists()

    def test_save_curation_uses_atomic_write(self, harness, tmp_path):
        """_save_curation must use _atomic_tiff_write, not raw tifffile."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True

        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write"
        ) as mock_atomic:
            harness._save_curation()
            mock_atomic.assert_called_once()


# ===========================================================================
# #5: Discard must revert in-memory volume from disk
# ===========================================================================

class TestDiscardRevertsVolume:
    """Choosing 'Discard' on unsaved-changes must reload volume from disk."""

    def test_revert_restores_on_disk_data(self, harness, tmp_path):
        """_revert_labels_from_disk reloads the TIFF into layer.data."""
        import tifffile

        # Write original to disk
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)

        # Simulate in-memory edit (split created label 5)
        layer.data[2, 3:6, 3:6] = 5

        harness._revert_labels_from_disk()

        np.testing.assert_array_equal(
            layer.data, original,
            err_msg="After revert, layer data must match the on-disk TIFF"
        )
        layer.refresh.assert_called()

    def test_revert_no_path_is_safe(self, harness, tmp_path):
        """_revert_labels_from_disk with no labels_path must not crash."""
        harness._labels_path = None
        harness._labels_layer = MagicMock()

        # Should not raise
        harness._revert_labels_from_disk()

    def test_revert_no_layer_is_safe(self, harness, tmp_path):
        """_revert_labels_from_disk with no labels layer must not crash."""
        harness._labels_path = str(tmp_path / "nonexistent.tif")
        harness._labels_layer = None

        harness._revert_labels_from_disk()

    def test_revert_missing_file_is_safe(self, harness, tmp_path):
        """_revert_labels_from_disk with missing TIFF must log warning."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        # Don't write any file to disk

        harness._revert_labels_from_disk()

        # Should not crash; file doesn't exist so no revert happens

    def test_revert_reloads_csv_features(self, harness, tmp_path):
        """_revert_labels_from_disk must also reload QC CSV from disk."""
        import tifffile

        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path)

        # Track whether _reload_current_patch_features is called
        reload_called = []
        original_reload = harness._reload_current_patch_features

        def tracking_reload(*a, **kw):
            reload_called.append(True)
            # Don't call original — it needs full UI setup
        harness._reload_current_patch_features = tracking_reload

        harness._revert_labels_from_disk()

        assert reload_called, (
            "_revert_labels_from_disk must call _reload_current_patch_features"
        )


# ===========================================================================
# #1: Stale curation_3d_ layer cleanup on annotator patch switch
# ===========================================================================

class TestStaleCurationLayerCleanup:
    """_switch_patch_from_3d_annotator must remove old curation_3d_ layers."""

    def test_clear_curation_layers_called_on_switch(self, harness, tmp_path):
        """Patch switch from annotator must call _clear_curation_layers."""
        _bind_real_methods(harness, [
            "_switch_patch_from_3d_annotator",
            "_switch_to_curation_patch",
            "_post_switch_annotator_sync",
            "_collect_flagged_labels",
            # 2026-06-06: the no-launch recovery in _switch_to_curation_patch
            # now sets the permanent status via _set_curation_status.
            "_set_curation_status",
        ])
        _setup_csv(harness, tmp_path, patch_name="patchA.tif")
        harness._qc_view_patch_name = "patchA.tif"

        # Set up state for patchB
        harness.state_manager._state["patches"]["patchB.tif"] = {
            "reconstruction_run_id": "run002",
        }

        clear_calls = []
        harness._clear_curation_layers = lambda: clear_calls.append(True)

        # Stub _guard_unsaved_changes to allow switch
        harness._guard_unsaved_changes = lambda: True

        harness._switch_patch_from_3d_annotator("patchB.tif")

        assert clear_calls, (
            "_clear_curation_layers must be called to remove stale "
            "curation_3d_ layers from old patch"
        )

    def test_same_patch_skips_cleanup(self, harness, tmp_path):
        """Same-patch guard must skip cleanup (no redundant layer removal)."""
        _bind_real_methods(harness, [
            "_switch_patch_from_3d_annotator",
            "_switch_to_curation_patch",
        ])
        harness._qc_view_patch_name = "patchA.tif"
        harness.state_manager._state["patches"]["patchA.tif"] = {
            "reconstruction_run_id": "run001",
        }

        clear_calls = []
        harness._clear_curation_layers = lambda: clear_calls.append(True)

        harness._switch_patch_from_3d_annotator("patchA.tif")

        assert not clear_calls, "Same-patch switch must not trigger cleanup"


# ===========================================================================
# #7: _reload_current_patch_features pw guard
# ===========================================================================

class TestReloadFeaturesPlotGuard:
    """Plot widget access must be inside the hasattr guard."""

    def test_no_qc_plot_widget_does_not_crash(self, harness, tmp_path):
        """If ui has no qc_plot_widget, _reload_current_patch_features
        must not crash with UnboundLocalError."""
        _setup_csv(harness, tmp_path)
        # Write a CSV to disk so load succeeds
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv
        csv_dir = tmp_path / "artifacts" / "step3" / "qc"
        csv_dir.mkdir(parents=True, exist_ok=True)
        save_patch_csv(harness._qc_features_df, str(csv_dir / "test_patch_features.csv"))

        # Remove qc_plot_widget from UI mock
        del harness.ui.qc_plot_widget

        # Must not raise UnboundLocalError
        harness._reload_current_patch_features()


# ===========================================================================
# #4: CSV/volume reconciliation — detect mismatch on load
# ===========================================================================

class TestCsvVolumeMismatchDetection:
    """When a CSV and volume disagree about which labels exist, the system
    should detect and log the inconsistency."""

    def test_csv_has_label_not_in_volume(self, harness, tmp_path):
        """CSV row for label 99 but label 99 not in volume → warning logged."""
        import tifffile

        # Volume only has labels 1, 2
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        data[2, 3:6, 3:6] = 2
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data)
        _write_initial_labels(labels_path, data)

        # CSV has labels 1, 2, 99 (99 doesn't exist in volume)
        df = pd.DataFrame({
            "label": [1, 2, 99],
            "area": [100, 200, 50],
            "qc_tier": ["good", "good", "review"],
            "feature_state": ["fresh", "fresh", "fresh"],
        })

        missing_in_vol = set(df["label"].astype(int)) - set(np.unique(data)) | {0}
        # Removing 0 from volume set
        vol_labels = set(np.unique(data)) - {0}
        csv_labels = set(df["label"].astype(int))

        missing_in_vol = csv_labels - vol_labels
        assert 99 in missing_in_vol, "Sanity: label 99 should be detected as missing"

    def test_volume_has_label_not_in_csv(self, harness, tmp_path):
        """Volume has label 3 but CSV has no row for it → detectable."""
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        data[2, 3:6, 3:6] = 3

        df = pd.DataFrame({
            "label": [1],
            "area": [100],
            "qc_tier": ["good"],
            "feature_state": ["fresh"],
        })

        vol_labels = set(np.unique(data)) - {0}
        csv_labels = set(df["label"].astype(int))

        missing_in_csv = vol_labels - csv_labels
        assert 3 in missing_in_csv, "Sanity: label 3 should be detected as missing"


# ===========================================================================
# Re-entry scenario: curate → commit → exit → relaunch → correct data
# ===========================================================================

class TestReentryDataIntegrity:
    """End-to-end test: simulate curate → commit → exit → reload cycle."""

    def test_curated_mask_survives_restart(self, harness, tmp_path):
        """After commit triage (no garbage), reloading from disk must
        return the curated mask, not the original reconstruction."""
        import tifffile

        # 1. Start with original reconstruction
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        original[2, 3:6, 3:6] = 2
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1, 2])

        # 2. Simulate curation: user splits label 2 into 3 and 4
        layer.data[layer.data == 2] = 0
        layer.data[2, 3:5, 3:5] = 3
        layer.data[2, 5:6, 5:6] = 4
        harness._curation_dirty = True

        # 3. Commit triage (no garbage)
        harness._commit_triage_single_patch()

        # 4. Verify mask on disk matches curated state
        reloaded = tifffile.imread(str(labels_path))
        assert 2 not in reloaded, "Old label 2 should not be in saved mask"
        assert 3 in reloaded, "Split fragment 3 must be in saved mask"
        assert 4 in reloaded, "Split fragment 4 must be in saved mask"
        assert 1 in reloaded, "Untouched label 1 must survive"

    def test_save_curation_mask_survives_restart(self, harness, tmp_path):
        """_save_curation must also persist curated data for re-entry."""
        import tifffile

        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1])

        # Simulate edit
        layer.data[3, 1:4, 1:4] = 5
        harness._curation_dirty = True

        harness._save_curation()

        assert harness._curation_dirty is False
        reloaded = tifffile.imread(str(labels_path))
        assert 5 in reloaded, "New label 5 must be persisted"

    def test_discard_then_commit_no_leaked_edits(self, harness, tmp_path):
        """After discard, commit triage must NOT contain discarded edits."""
        import tifffile

        # Original on disk
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1])

        # Simulate edit (user draws label 99)
        layer.data[3, 6:8, 6:8] = 99
        harness._curation_dirty = True

        # Discard via revert
        harness._revert_labels_from_disk()
        harness._curation_dirty = False

        # Now commit triage — should only have original data
        harness._curation_dirty = False
        harness._tiers_dirty = True
        _setup_csv(harness, tmp_path, labels=[1])
        harness._commit_triage_single_patch()

        reloaded = tifffile.imread(str(labels_path))
        assert 99 not in reloaded, (
            "Discarded label 99 must NOT appear in committed mask"
        )
        assert 1 in reloaded

    def test_commit_triage_clears_both_dirty_flags(self, harness, tmp_path):
        """Both _curation_dirty and _tiers_dirty must be False after commit."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        harness._tiers_dirty = True

        harness._commit_triage_single_patch()

        assert harness._curation_dirty is False
        assert harness._tiers_dirty is False

    def test_save_curation_then_load_roundtrips(self, harness, tmp_path):
        """Save → reload from disk must produce identical data."""
        import tifffile

        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 7
        data[2, 3:6, 3:6] = 13
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        _setup_csv(harness, tmp_path, labels=[7, 13])
        harness._curation_dirty = True

        harness._save_curation()

        reloaded = tifffile.imread(str(labels_path))
        np.testing.assert_array_equal(reloaded, data)


# ===========================================================================
# R4.3.1-1: _commit_triage_single_patch must NOT clear dirty flags on
#            partial persistence failure
# ===========================================================================

class TestCommitTriageDirtyFlagOnFailure:
    """_commit_triage_single_patch must keep dirty flags True if any
    persistence step (mask or CSV) fails."""

    def test_mask_write_failure_keeps_curation_dirty(self, harness, tmp_path):
        """If label-volume save raises, _curation_dirty must stay True."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        # Sabotage: make _atomic_tiff_write raise
        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            harness._commit_triage_single_patch()

        assert harness._curation_dirty is True, (
            "Mask write failed → _curation_dirty must remain True"
        )

    def test_mask_write_failure_keeps_tiers_dirty(self, harness, tmp_path):
        """If label-volume save raises, _tiers_dirty must stay True too
        (the commit is not complete)."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            harness._commit_triage_single_patch()

        assert harness._tiers_dirty is True, (
            "Mask write failed → _tiers_dirty must remain True"
        )

    def test_csv_write_failure_keeps_dirty(self, harness, tmp_path):
        """If CSV save raises, both dirty flags must stay True."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        with patch.object(
            pd.DataFrame, "to_csv",
            side_effect=OSError("permission denied"),
        ):
            harness._commit_triage_single_patch()

        assert harness._curation_dirty is True, (
            "CSV write failed → _curation_dirty must remain True"
        )
        assert harness._tiers_dirty is True, (
            "CSV write failed → _tiers_dirty must remain True"
        )

    def test_mask_write_failure_logs_error(self, harness, tmp_path):
        """Failed mask write must produce an explicit error log."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            harness._commit_triage_single_patch()

        error_logs = [m for m in harness._log if "[ERROR]" in m]
        assert error_logs, "Mask write failure must produce [ERROR] log entry"

    def test_success_still_clears_both_flags(self, harness, tmp_path):
        """When both writes succeed, both dirty flags must be cleared."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        harness._commit_triage_single_patch()

        assert harness._curation_dirty is False
        assert harness._tiers_dirty is False


# ===========================================================================
# R4.3.1-2: _save_curation transactional — rollback on partial failure
# ===========================================================================

class TestSaveCurationTransactional:
    """_save_curation must not leave mask updated + CSV stale (or vice versa).
    On partial failure, the successful write must be rolled back."""

    def test_csv_failure_rolls_back_mask(self, harness, tmp_path):
        """If mask writes OK but CSV fails, mask must be rolled back to
        its pre-save state on disk."""
        import tifffile

        # Write original mask to disk
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        # Modify in-memory data
        layer.data[3, 6:8, 6:8] = 99

        # Sabotage CSV write
        with patch(
            "napari_epicycle.fusion_reconstruction.qc_features.save_patch_csv",
            side_effect=OSError("permission denied"),
        ):
            harness._save_curation()

        # Mask on disk must still be the original — rolled back
        on_disk = tifffile.imread(str(labels_path))
        np.testing.assert_array_equal(on_disk, original, err_msg=(
            "Mask must be rolled back to original when CSV write fails"
        ))

    def test_mask_failure_does_not_write_csv(self, harness, tmp_path):
        """If mask write fails, CSV must NOT be written either."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True

        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "test_patch_features.csv"
        assert not csv_path.exists(), "Precondition: no CSV on disk yet"

        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            harness._save_curation()

        assert not csv_path.exists(), (
            "CSV must NOT be written when mask write fails"
        )
        assert harness._curation_dirty is True

    def test_both_succeed_both_written(self, harness, tmp_path):
        """When both succeed, both files must exist on disk."""
        import tifffile

        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        harness._save_curation()

        assert labels_path.exists()
        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "test_patch_features.csv"
        assert csv_path.exists()
        assert harness._curation_dirty is False

    def test_partial_failure_keeps_dirty(self, harness, tmp_path):
        """Any failure must keep _curation_dirty=True."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            harness._save_curation()

        assert harness._curation_dirty is True

    def test_rollback_logged_on_failure(self, harness, tmp_path):
        """Rollback attempt must be logged."""
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        layer.data[3, 6:8, 6:8] = 99

        with patch(
            "napari_epicycle.fusion_reconstruction.qc_features.save_patch_csv",
            side_effect=OSError("permission denied"),
        ):
            harness._save_curation()

        error_logs = [m for m in harness._log if "[ERROR]" in m]
        assert error_logs, "Partial failure must log [ERROR] with rollback info"


# ===========================================================================
# R4.3.1-3: CSV/volume mismatch — real controller-level behavior
# ===========================================================================

class TestCsvVolumeMismatchControllerBehavior:
    """Controller must detect and log CSV/volume label mismatches when
    loading QC features for a patch that has a labels layer in the viewer."""

    def test_csv_labels_missing_from_volume_logs_warning(self, harness, tmp_path):
        """CSV has label IDs not present in volume → controller logs warning."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv
        # _reload_current_patch_features + _check_csv_volume_consistency
        # are already bound by the harness fixture.

        # Volume has labels 1, 2
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        data[2, 3:6, 3:6] = 2
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data)

        # CSV has labels 1, 2, 99
        csv_dir = tmp_path / "artifacts" / "step3" / "qc"
        csv_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "label": [1, 2, 99],
            "area": [100, 200, 50],
            "qc_tier": ["good", "good", "review"],
            "feature_state": ["fresh", "fresh", "fresh"],
        })
        save_patch_csv(df, str(csv_dir / "test_patch_features.csv"))
        harness._cur_patch = "test_patch.tif"
        harness._patch_name = "test_patch.tif"

        harness._reload_current_patch_features()

        warn_logs = [m for m in harness._log if "mismatch" in m.lower()]
        assert warn_logs, (
            "CSV labels not in volume must produce a mismatch warning log"
        )

    def test_volume_labels_missing_from_csv_logs_warning(self, harness, tmp_path):
        """Volume has label IDs not in CSV → controller logs warning."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv
        # _reload_current_patch_features + _check_csv_volume_consistency
        # are already bound by the harness fixture.

        # Volume has labels 1, 2, 3
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        data[2, 3:6, 3:6] = 2
        data[3, 1:3, 1:3] = 3
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data)

        # CSV only has label 1
        csv_dir = tmp_path / "artifacts" / "step3" / "qc"
        csv_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "label": [1],
            "area": [100],
            "qc_tier": ["good"],
            "feature_state": ["fresh"],
        })
        save_patch_csv(df, str(csv_dir / "test_patch_features.csv"))
        harness._cur_patch = "test_patch.tif"
        harness._patch_name = "test_patch.tif"

        harness._reload_current_patch_features()

        warn_logs = [m for m in harness._log if "mismatch" in m.lower()]
        assert warn_logs, (
            "Volume labels not in CSV must produce a mismatch warning log"
        )

    def test_matching_labels_no_warning(self, harness, tmp_path):
        """When CSV and volume labels match, no mismatch warning is logged."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv
        # _reload_current_patch_features + _check_csv_volume_consistency
        # are already bound by the harness fixture.

        # Volume has labels 1, 2
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        data[2, 3:6, 3:6] = 2
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data)

        # CSV also has labels 1, 2
        csv_dir = tmp_path / "artifacts" / "step3" / "qc"
        csv_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "label": [1, 2],
            "area": [100, 200],
            "qc_tier": ["good", "good"],
            "feature_state": ["fresh", "fresh"],
        })
        save_patch_csv(df, str(csv_dir / "test_patch_features.csv"))
        harness._cur_patch = "test_patch.tif"
        harness._patch_name = "test_patch.tif"

        harness._reload_current_patch_features()

        warn_logs = [m for m in harness._log if "mismatch" in m.lower()]
        assert not warn_logs, "No mismatch warning when labels match"

    def test_mismatch_does_not_crash(self, harness, tmp_path):
        """Mismatch must not raise — features load proceeds normally."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv
        # _reload_current_patch_features + _check_csv_volume_consistency
        # are already bound by the harness fixture.

        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data)

        csv_dir = tmp_path / "artifacts" / "step3" / "qc"
        csv_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "label": [1, 50, 99],
            "area": [100, 200, 50],
            "qc_tier": ["good", "good", "review"],
            "feature_state": ["fresh", "fresh", "fresh"],
        })
        save_patch_csv(df, str(csv_dir / "test_patch_features.csv"))
        harness._cur_patch = "test_patch.tif"
        harness._patch_name = "test_patch.tif"

        # Must not raise
        harness._reload_current_patch_features()

        # DataFrame must still be loaded
        assert harness._qc_features_df is not None
        assert len(harness._qc_features_df) == 3


# ===========================================================================
# R4.3.2-1: Mask-only save when CSV target is unavailable
# ===========================================================================

class TestMaskOnlySave:
    """_save_curation must write mask even when CSV target is missing."""

    def test_mask_saved_without_qc_df(self, harness, tmp_path):
        """With valid mask layer/path but no _qc_features_df,
        mask must still be written to disk."""
        import tifffile

        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        harness._qc_features_df = None  # No CSV target
        harness._cur_patch = "test_patch.tif"
        harness._patch_name = "test_patch.tif"
        harness._curation_dirty = True

        harness._save_curation()

        assert labels_path.exists(), "Mask must be saved even without QC DataFrame"
        reloaded = tifffile.imread(str(labels_path))
        np.testing.assert_array_equal(reloaded, data)

    def test_mask_only_clears_dirty(self, harness, tmp_path):
        """Mask-only save must clear _curation_dirty on success."""
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        harness._qc_features_df = None
        harness._curation_dirty = True

        harness._save_curation()

        assert harness._curation_dirty is False, (
            "Mask-only save must clear dirty flag on success"
        )

    def test_mask_only_logs_csv_skipped(self, harness, tmp_path):
        """Mask-only save must log why CSV was skipped."""
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        harness._qc_features_df = None
        harness._curation_dirty = True

        harness._save_curation()

        csv_skip_logs = [m for m in harness._log if "csv" in m.lower() and "skip" in m.lower()]
        assert csv_skip_logs, "Must log that CSV was skipped"

    def test_mask_only_failure_keeps_dirty(self, harness, tmp_path):
        """If mask-only write fails, dirty flag must stay True."""
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        harness._qc_features_df = None
        harness._curation_dirty = True

        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            harness._save_curation()

        assert harness._curation_dirty is True


# ===========================================================================
# R4.3.2-2: Low-memory rollback — no read_bytes on mask
# ===========================================================================

class TestLowMemoryRollback:
    """_save_curation must NOT load the entire mask into memory for rollback.
    Instead it must use a rename/backup-file strategy."""

    def test_no_read_bytes_in_save_curation(self, harness, tmp_path):
        """_save_curation must not call Path.read_bytes() for backup."""
        import tifffile

        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        read_bytes_calls = []
        _orig_read_bytes = Path.read_bytes

        def tracking_read_bytes(self_path):
            read_bytes_calls.append(str(self_path))
            return _orig_read_bytes(self_path)

        with patch.object(Path, "read_bytes", tracking_read_bytes):
            harness._save_curation()

        assert len(read_bytes_calls) == 0, (
            f"read_bytes was called on: {read_bytes_calls}. "
            "Must use rename-based rollback, not in-memory backup."
        )

    def test_csv_failure_original_untouched(self, harness, tmp_path):
        """On CSV failure, original mask must remain unchanged on disk
        (stage-then-promote: original is never moved)."""
        import tifffile

        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        layer.data[3, 6:8, 6:8] = 99

        with patch(
            "napari_epicycle.fusion_reconstruction.qc_features.save_patch_csv",
            side_effect=OSError("permission denied"),
        ):
            harness._save_curation()

        # Original must be untouched — staging was deleted
        on_disk = tifffile.imread(str(labels_path))
        np.testing.assert_array_equal(on_disk, original)

        # No staging file left behind
        staging_files = list(labels_path.parent.glob("*.saving"))
        assert len(staging_files) == 0

    def test_no_staging_file_left_on_success(self, harness, tmp_path):
        """After successful save, no staging file must remain."""
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        _write_initial_labels(labels_path, data)
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        harness._save_curation()

        staging_files = list(labels_path.parent.glob("*.saving"))
        assert len(staging_files) == 0, f"Staging files left behind: {staging_files}"


# ===========================================================================
# R4.3.2-3: Triage state metadata deferred until persistence succeeds
# ===========================================================================

class TestTriageStateOrdering:
    """state_manager.set_patch_status(qc_triage=...) must only be called
    AFTER both CSV and mask persistence succeed."""

    def test_mask_failure_no_state_update(self, harness, tmp_path):
        """If mask write fails, set_patch_status must NOT be called."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            harness._commit_triage_single_patch()

        # set_patch_status must NOT have been called with qc_triage
        for call in harness.state_manager.set_patch_status.call_args_list:
            _, kwargs = call
            assert "qc_triage" not in kwargs, (
                "set_patch_status(qc_triage=...) must not be called on mask failure"
            )

    def test_csv_failure_no_state_update(self, harness, tmp_path):
        """If CSV write fails, set_patch_status must NOT be called."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        with patch.object(
            pd.DataFrame, "to_csv",
            side_effect=OSError("permission denied"),
        ):
            harness._commit_triage_single_patch()

        for call in harness.state_manager.set_patch_status.call_args_list:
            _, kwargs = call
            assert "qc_triage" not in kwargs, (
                "set_patch_status(qc_triage=...) must not be called on CSV failure"
            )

    def test_success_commits_state(self, harness, tmp_path):
        """On full success, set_patch_status(qc_triage=...) must be called."""
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        harness._commit_triage_single_patch()

        triage_calls = [
            call for call in harness.state_manager.set_patch_status.call_args_list
            if call[1].get("qc_triage") is not None
        ]
        assert triage_calls, (
            "set_patch_status(qc_triage=...) must be called on success"
        )


# ===========================================================================
# R4.3.3-1: Atomic CSV writes via save_patch_csv
# ===========================================================================

class TestAtomicCsvWrite:
    """save_patch_csv must write atomically — no partial CSV on crash."""

    def test_csv_write_failure_preserves_original(self, tmp_path):
        """If to_csv raises mid-write (partial corruption), original CSV
        must remain intact because save_patch_csv uses atomic writes."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv

        csv_path = tmp_path / "features.csv"
        original_df = pd.DataFrame({"label": [1, 2], "area": [100, 200]})
        save_patch_csv(original_df, str(csv_path))

        # Simulate partial write: to_csv writes garbage then raises.
        _real_to_csv = pd.DataFrame.to_csv

        def partial_write(self_df, path_or_buf, **kwargs):
            # Write partial corrupt content to the path, then fail.
            with open(path_or_buf, "w") as f:
                f.write("CORRUPT_PARTIAL")
            raise OSError("disk full after partial write")

        bad_df = pd.DataFrame({"label": [10, 20], "area": [999, 888]})
        with patch.object(pd.DataFrame, "to_csv", partial_write):
            with pytest.raises(OSError):
                save_patch_csv(bad_df, str(csv_path))

        # Original must still be intact (not corrupted)
        reloaded = pd.read_csv(str(csv_path))
        assert list(reloaded["label"]) == [1, 2]

    def test_no_temp_csv_left_on_failure(self, tmp_path):
        """Failed CSV write must not leave temp files behind."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv

        csv_path = tmp_path / "features.csv"

        def partial_write(self_df, path_or_buf, **kwargs):
            with open(path_or_buf, "w") as f:
                f.write("CORRUPT")
            raise OSError("disk full")

        with patch.object(pd.DataFrame, "to_csv", partial_write):
            with pytest.raises(OSError):
                save_patch_csv(
                    pd.DataFrame({"label": [1]}), str(csv_path)
                )

        all_files = list(tmp_path.glob("*"))
        assert len(all_files) == 0, f"Temp/residual files left behind: {all_files}"

    def test_successful_atomic_csv_write(self, tmp_path):
        """Normal write must produce correct output."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv

        csv_path = tmp_path / "features.csv"
        df = pd.DataFrame({"label": [1, 2, 3], "area": [10, 20, 30]})
        save_patch_csv(df, str(csv_path))

        reloaded = pd.read_csv(str(csv_path))
        assert list(reloaded["label"]) == [1, 2, 3]

    def test_overwrites_existing_atomically(self, tmp_path):
        """Overwriting an existing CSV must succeed without corruption."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv

        csv_path = tmp_path / "features.csv"
        save_patch_csv(pd.DataFrame({"label": [1]}), str(csv_path))
        save_patch_csv(pd.DataFrame({"label": [99]}), str(csv_path))

        reloaded = pd.read_csv(str(csv_path))
        assert list(reloaded["label"]) == [99]


# ===========================================================================
# R4.3.3-2: Triage commit — no mask/CSV divergence on partial failure
# ===========================================================================

class TestTriageNoDivergence:
    """_commit_triage_single_patch must not write mask if CSV fails,
    and must not leave CSV promoted if mask fails."""

    def test_csv_failure_mask_unchanged(self, harness, tmp_path):
        """If CSV save fails, mask file must remain unchanged on disk."""
        import tifffile

        # Write original mask to disk
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._qc_features_df["qc_tier"] = ["good", "garbage"]
        harness._curation_dirty = True
        harness._tiers_dirty = True

        def _zero(ids):
            for lid in ids:
                layer.data[layer.data == lid] = 0
        harness._zero_labels_in_viewer = _zero

        with patch(
            "napari_epicycle.fusion_reconstruction.qc_features.save_patch_csv",
            side_effect=OSError("permission denied"),
        ):
            harness._commit_triage_single_patch()

        # Mask on disk must still be the original
        on_disk = tifffile.imread(str(labels_path))
        np.testing.assert_array_equal(on_disk, original, err_msg=(
            "Mask must not be written when CSV save fails"
        ))

    def test_mask_failure_csv_unchanged(self, harness, tmp_path):
        """If mask save fails after CSV staged, CSV must not be promoted."""
        from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv

        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        # Write a pre-existing CSV to disk
        csv_dir = tmp_path / "artifacts" / "step3" / "qc"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "test_patch_features.csv"
        original_csv = pd.DataFrame({
            "label": [1, 2],
            "area": [100, 200],
            "qc_tier": ["unset", "unset"],
            "feature_state": ["fresh", "fresh"],
        })
        save_patch_csv(original_csv, str(csv_path))

        # Now triage assigns tiers
        harness._qc_features_df["qc_tier"] = ["good", "review"]

        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            harness._commit_triage_single_patch()

        # CSV on disk must still have original content
        reloaded = pd.read_csv(str(csv_path))
        assert list(reloaded["qc_tier"]) == ["unset", "unset"], (
            "CSV must not be updated when mask save fails"
        )

    def test_both_succeed_both_updated(self, harness, tmp_path):
        """When both succeed, both files must reflect the new state."""
        import tifffile

        data = np.zeros((4, 8, 8), dtype=np.uint32)
        data[1, 2:5, 2:5] = 1
        data[2, 3:6, 3:6] = 2
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        _setup_csv(harness, tmp_path, labels=[1, 2])
        harness._curation_dirty = True
        harness._tiers_dirty = True

        harness._commit_triage_single_patch()

        assert labels_path.exists()
        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "test_patch_features.csv"
        assert csv_path.exists()
        assert harness._curation_dirty is False
        assert harness._tiers_dirty is False


# ===========================================================================
# R4.3.3-3: Promote failure cleanup in _save_curation
# ===========================================================================

class TestPromoteFailureCleanup:
    """On promote failure, _save_curation must clean up the staging file
    and leave the filesystem in a deterministic state."""

    def test_promote_failure_cleans_staging(self, harness, tmp_path):
        """On os.replace failure, staging file must be deleted."""
        import tifffile

        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=original.copy())
        _write_initial_labels(labels_path, original)
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        # Patch os.replace to fail only on the promote call
        _real_replace = os.replace
        promote_count = [0]

        def failing_replace(src, dst):
            # _atomic_tiff_write uses os.replace internally for its
            # temp→staging rename. Let that one through, fail the
            # staging→labels_path promote.
            if str(dst) == str(labels_path):
                promote_count[0] += 1
                raise OSError("promote failed")
            return _real_replace(src, dst)

        with patch("os.replace", side_effect=failing_replace):
            harness._save_curation()

        assert promote_count[0] > 0, "Sanity: promote must have been attempted"

        # Staging file must be cleaned up
        staging_files = list(labels_path.parent.glob("*.saving"))
        assert len(staging_files) == 0, (
            f"Staging file must be cleaned up on promote failure: {staging_files}"
        )

        # Original must be untouched
        on_disk = tifffile.imread(str(labels_path))
        np.testing.assert_array_equal(on_disk, original)

        # Dirty flag must remain True
        assert harness._curation_dirty is True

    def test_promote_failure_logs_cleanup_result(self, harness, tmp_path):
        """Promote failure must log the cleanup outcome explicitly."""
        data = np.zeros((4, 8, 8), dtype=np.uint32)
        layer, labels_path = _setup_labels(harness, tmp_path, label_data=data.copy())
        _write_initial_labels(labels_path, data)
        _setup_csv(harness, tmp_path, labels=[1])
        harness._curation_dirty = True

        _real_replace = os.replace

        def failing_replace(src, dst):
            if str(dst) == str(labels_path):
                raise OSError("promote failed")
            return _real_replace(src, dst)

        with patch("os.replace", side_effect=failing_replace):
            harness._save_curation()

        error_logs = [m for m in harness._log if "[ERROR]" in m]
        assert error_logs, "Promote failure must produce [ERROR] log"
        # Must mention cleanup result
        cleanup_logs = [m for m in harness._log if "clean" in m.lower()]
        assert cleanup_logs, "Must log staging cleanup result"


# ===========================================================================
# 2026-06-07: Commit-Triage went ASYNC (background QThread persist).  The
# transactional persist is a PURE function (_commit_triage_persist) that BOTH
# the worker and these tests call synchronously; the in-memory state update +
# dirty-flag handling live in the main-thread handlers
# (_on_commit_triage_finished / _on_commit_triage_error).  The tests below
# drive that synchronous path directly (no live QThread).
# ===========================================================================


class TestCommitTriagePersistPure:
    """The PURE _commit_triage_persist holds the transactional staging
    (CSV stage → mask write → CSV promote) verbatim — stage failures leave no
    orphan temp file, promote happens only after the mask write."""

    def _make_kwargs(self, tmp_path, *, garbage_ids=None, mask=None,
                     labels_name="labels_3d.tif"):
        import numpy as np
        csv_dir = tmp_path / "qc"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "patch_features.csv"
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        labels_path = labels_dir / labels_name
        df = pd.DataFrame({
            "label": [1, 2],
            "area": [100, 200],
            "qc_tier": ["good", "review"],
            "label_status": ["active", "active"],
        })
        if mask is None:
            mask = np.zeros((4, 8, 8), dtype=np.uint32)
            mask[1, 2:5, 2:5] = 1
        return dict(
            df=df,
            garbage_ids=garbage_ids or set(),
            lifted_mask=mask,
            labels_path=str(labels_path),
            csv_path=csv_path,
            csv_dir=csv_dir,
        ), csv_path, labels_path

    def test_success_writes_both_and_builds_triage_map(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _commit_triage_persist,
        )
        import tifffile

        kwargs, csv_path, labels_path = self._make_kwargs(tmp_path)
        result = _commit_triage_persist(**kwargs)

        assert result["ok"] is True
        assert result["stage"] == "done"
        # Both files written.
        assert labels_path.exists()
        assert csv_path.exists()
        # Triage map covers both active labels.
        assert result["triage_map"]["1"] == "good"
        assert result["triage_map"]["2"] == "review"
        # No orphan staging temp file in csv_dir.
        assert list(kwargs["csv_dir"].glob("tmp*")) == []

    def test_garbage_in_triage_map_as_discarded(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _commit_triage_persist,
        )
        kwargs, csv_path, labels_path = self._make_kwargs(
            tmp_path, garbage_ids={9}
        )
        result = _commit_triage_persist(**kwargs)
        assert result["ok"] is True
        assert result["triage_map"]["9"] == "discarded"

    def test_csv_stage_failure_no_mask_no_orphan(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _commit_triage_persist,
        )
        kwargs, csv_path, labels_path = self._make_kwargs(tmp_path)
        with patch.object(
            pd.DataFrame, "to_csv", side_effect=OSError("disk full")
        ):
            result = _commit_triage_persist(**kwargs)
        assert result["ok"] is False
        assert result["stage"] == "csv_stage"
        # Mask NOT written (stage failed before mask).
        assert not labels_path.exists()
        # No orphan staging temp file.
        assert list(kwargs["csv_dir"].glob("tmp*")) == []

    def test_mask_failure_discards_staged_csv_no_promote(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _commit_triage_persist,
        )
        kwargs, csv_path, labels_path = self._make_kwargs(tmp_path)
        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            result = _commit_triage_persist(**kwargs)
        assert result["ok"] is False
        assert result["stage"] == "mask"
        # CSV NOT promoted.
        assert not csv_path.exists()
        # Staged CSV discarded — no orphan.
        assert list(kwargs["csv_dir"].glob("tmp*")) == []

    def test_mask_skipped_when_no_mask(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _commit_triage_persist,
        )
        kwargs, csv_path, labels_path = self._make_kwargs(tmp_path, mask=None)
        kwargs["lifted_mask"] = None  # no mask available
        result = _commit_triage_persist(**kwargs)
        assert result["ok"] is False
        assert result["mask_skipped"] is True
        # CSV not promoted, no orphan.
        assert not csv_path.exists()
        assert list(kwargs["csv_dir"].glob("tmp*")) == []

    def test_promote_failure_discards_staged_csv(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _commit_triage_persist,
        )
        kwargs, csv_path, labels_path = self._make_kwargs(tmp_path)
        _real_replace = os.replace

        def failing_replace(src, dst):
            # Let _atomic_tiff_write's internal temp→labels rename through;
            # fail only the CSV staging→csv_path promote.
            if str(dst) == str(csv_path):
                raise OSError("promote failed")
            return _real_replace(src, dst)

        with patch("os.replace", side_effect=failing_replace):
            result = _commit_triage_persist(**kwargs)
        assert result["ok"] is False
        assert result["stage"] == "promote"
        # Mask WAS written (promote is the last step).
        assert labels_path.exists()
        # Staged CSV discarded — no orphan, and final CSV not present.
        assert not csv_path.exists()
        assert list(kwargs["csv_dir"].glob("tmp*")) == []


class TestCommitTriageReentryGate:
    """_commit_triage_in_flight must gate re-entry: a click while a commit is
    already running is ignored (no prep, no new launch)."""

    def test_in_flight_blocks_reentry(self, harness, tmp_path):
        layer, labels_path = _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        # Simulate a commit already in flight.
        harness._commit_triage_in_flight = True

        # Track whether prep runs (it would on a non-gated entry).
        prep_calls = []
        _real_prep = harness._commit_triage_prepare

        def tracking_prep():
            prep_calls.append(True)
            return _real_prep()
        harness._commit_triage_prepare = tracking_prep

        harness._commit_triage()

        assert not prep_calls, (
            "_commit_triage must early-return without running prep when a "
            "commit is already in flight"
        )
        # Dirty flag untouched (no commit ran).
        assert harness._curation_dirty is True


class TestCommitTriageStaleFinish:
    """A stale finish (token bumped or active patch drifted after a switch
    that slipped the lock) must NOT clobber the new patch's in-memory state."""

    def test_stale_token_skips_state_update(self, harness, tmp_path):
        _setup_csv(harness, tmp_path)
        # The NEW patch (after a switch) has its own dirty state.
        harness._curation_dirty = True
        harness._tiers_dirty = True
        harness._qc_view_patch_name = "new_patch.tif"
        # Current token is 5; the in-flight commit dispatched at token 4.
        harness._commit_triage_token = 5
        harness._commit_triage_in_flight = True

        result = {
            "ok": True,
            "stage": "done",
            "triage_map": {"1": "good"},
            "n_labels": 1,
        }
        # Finish for the SUPERSEDED dispatch (token 4, old patch).
        harness._on_commit_triage_finished(
            result, 4, "old_patch.tif",
            {"garbage_ids": set(), "labels_path": None},
        )

        # In-memory state NOT clobbered — the new patch's dirty flags survive.
        assert harness._curation_dirty is True
        assert harness._tiers_dirty is True
        # set_patch_status(qc_triage=...) must NOT have fired for the old patch.
        for call in harness.state_manager.set_patch_status.call_args_list:
            _, kwargs = call
            assert kwargs.get("qc_triage") is None, (
                "stale finish must not commit the superseded patch's triage"
            )
        # In-flight flag cleared + bar hidden (finally block runs).
        assert harness._commit_triage_in_flight is False

    def test_patch_drift_skips_state_update(self, harness, tmp_path):
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        harness._tiers_dirty = True
        # Token matches, but the active QC-view patch drifted post-dispatch.
        harness._commit_triage_token = 3
        harness._qc_view_patch_name = "patchB.tif"
        harness._commit_triage_in_flight = True

        result = {"ok": True, "stage": "done",
                  "triage_map": {"1": "good"}, "n_labels": 1}
        harness._on_commit_triage_finished(
            result, 3, "patchA.tif",
            {"garbage_ids": set(), "labels_path": None},
        )

        assert harness._curation_dirty is True
        assert harness._tiers_dirty is True
        assert harness._commit_triage_in_flight is False


class TestCommitTriageHandlerInvariants:
    """_on_commit_triage_finished / _on_commit_triage_error reproduce the
    legacy transactional branches: dirty flags stay TRUE on any failure,
    set_patch_status(qc_triage=...) only on full success."""

    def test_ok_false_keeps_dirty_no_state(self, harness, tmp_path):
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        harness._tiers_dirty = True
        harness._commit_triage_in_flight = True
        harness._qc_view_patch_name = "test_patch.tif"

        result = {
            "ok": False,
            "stage": "mask",
            "error": "disk full",
            "triage_map": {"1": "good"},
            "n_labels": 1,
        }
        harness._on_commit_triage_finished(
            result, harness._commit_triage_token, "test_patch.tif",
            {"garbage_ids": set(), "labels_path": "labels.tif"},
        )

        assert harness._curation_dirty is True
        assert harness._tiers_dirty is True
        for call in harness.state_manager.set_patch_status.call_args_list:
            _, kwargs = call
            assert kwargs.get("qc_triage") is None
        assert harness._commit_triage_in_flight is False

    def test_ok_true_clears_dirty_commits_state(self, harness, tmp_path):
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        harness._tiers_dirty = True
        harness._commit_triage_in_flight = True
        harness._qc_view_patch_name = "test_patch.tif"

        result = {
            "ok": True,
            "stage": "done",
            "triage_map": {"1": "good", "2": "review"},
            "n_labels": 2,
        }
        harness._on_commit_triage_finished(
            result, harness._commit_triage_token, "test_patch.tif",
            {"garbage_ids": set(), "labels_path": "labels.tif"},
        )

        assert harness._curation_dirty is False
        assert harness._tiers_dirty is False
        triage_calls = [
            c for c in harness.state_manager.set_patch_status.call_args_list
            if c[1].get("qc_triage") is not None
        ]
        assert triage_calls, "full success must commit qc_triage"
        assert triage_calls[0][1]["qc_triage"] == {"1": "good", "2": "review"}
        assert harness._commit_triage_in_flight is False

    def test_worker_error_keeps_dirty(self, harness, tmp_path):
        harness._curation_dirty = True
        harness._tiers_dirty = True
        harness._commit_triage_in_flight = True

        harness._on_commit_triage_error(
            "boom\ntraceback...", harness._commit_triage_token, "test_patch.tif"
        )

        assert harness._curation_dirty is True
        assert harness._tiers_dirty is True
        assert harness._commit_triage_in_flight is False
        error_logs = [m for m in harness._log if "[ERROR]" in m]
        assert error_logs, "worker error must log [ERROR]"


# ===========================================================================
# 2026-06-07: Save Curation went ASYNC (Save BUTTON → background QThread
# persist).  The transactional persist is a PURE function
# (_save_curation_persist) that BOTH the worker and the SYNCHRONOUS
# orchestrator (_save_curation) / these tests call; the in-memory dirty-flag
# handling lives in the main-thread handlers (_on_save_curation_finished /
# _on_save_curation_error).  The tests below drive that path directly (no
# live QThread), mirroring the commit-triage async tests.
# ===========================================================================


class TestSaveCurationPersistPure:
    """The PURE _save_curation_persist holds the stage→.bak→promote transaction
    verbatim — stage failures leave no orphan staging file, the original is
    never touched until the final promote, and .bak rotation is a COPY on the
    success path only."""

    def _kwargs(self, tmp_path, *, mask=None, csv_df=None, csv_path=None,
                labels_path=None):
        if mask is None:
            mask = np.full((4, 8, 8), 7, dtype=np.uint32)
        if labels_path is None:
            labels_dir = tmp_path / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            labels_path = labels_dir / "labels_3d.tif"
        from napari_epicycle.fusion_reconstruction.qc_features import (
            save_patch_csv,
        )
        return {
            "lifted_mask": mask,
            "labels_path": str(labels_path),
            "csv_df": csv_df,
            "csv_path": str(csv_path) if csv_path is not None else None,
            "save_patch_csv": save_patch_csv,
        }, Path(labels_path)

    def test_mask_only_success_writes_and_promotes(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _save_curation_persist,
        )
        import tifffile
        kwargs, labels_path = self._kwargs(tmp_path)  # csv_df None → mask-only
        res = _save_curation_persist(**kwargs)
        assert res["ok"] is True
        assert res["mask_only"] is True
        assert res["stage"] == "done"
        assert labels_path.exists()
        np.testing.assert_array_equal(
            tifffile.imread(str(labels_path)),
            np.full((4, 8, 8), 7, dtype=np.uint32),
        )
        # No orphaned staging file.
        assert not list(labels_path.parent.glob("*.saving"))

    def test_mask_and_csv_success(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            _save_curation_persist,
        )
        import tifffile
        csv_path = tmp_path / "feat.csv"
        df = pd.DataFrame({"label": [1, 2], "area": [10, 20]})
        kwargs, labels_path = self._kwargs(
            tmp_path, csv_df=df, csv_path=csv_path
        )
        res = _save_curation_persist(**kwargs)
        assert res["ok"] is True
        assert res["mask_only"] is False
        assert labels_path.exists()
        assert csv_path.exists()
        assert tifffile.imread(str(labels_path)).shape == (4, 8, 8)

    def test_mask_stage_failure_no_orphan_no_csv(self, tmp_path):
        """If the mask stage write fails, no staging file remains, CSV not
        written, original (if any) untouched."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _save_curation_persist,
        )
        csv_path = tmp_path / "feat.csv"
        df = pd.DataFrame({"label": [1]})
        kwargs, labels_path = self._kwargs(
            tmp_path, csv_df=df, csv_path=csv_path
        )
        with patch(
            "napari_epicycle.fusion_reconstruction.controller._atomic_tiff_write",
            side_effect=OSError("disk full"),
        ):
            res = _save_curation_persist(**kwargs)
        assert res["ok"] is False
        assert res["stage"] == "mask_stage"
        assert not csv_path.exists(), "CSV must NOT be written on mask stage failure"
        assert not list(labels_path.parent.glob("*.saving"))

    def test_csv_failure_discards_staging_original_untouched(self, tmp_path):
        """CSV write failure → staging file deleted, original mask untouched,
        no promote (rollback substrate is the staging file)."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _save_curation_persist,
        )
        import tifffile
        # Pre-existing original on disk.
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        labels_path = labels_dir / "labels_3d.tif"
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[1, 2:5, 2:5] = 1
        tifffile.imwrite(str(labels_path), original, bigtiff=True)

        csv_path = tmp_path / "feat.csv"
        df = pd.DataFrame({"label": [1]})
        kwargs, _ = self._kwargs(
            tmp_path, mask=np.full((4, 8, 8), 9, dtype=np.uint32),
            csv_df=df, csv_path=csv_path, labels_path=labels_path,
        )
        # Inject a failing save_patch_csv (the pure fn calls the INJECTED
        # callable, so we override it directly here — the same way prep injects
        # the real qc_features.save_patch_csv).
        def _boom(df_, path_):
            raise OSError("permission denied")
        kwargs["save_patch_csv"] = _boom
        res = _save_curation_persist(**kwargs)
        assert res["ok"] is False
        assert res["stage"] == "csv"
        # Original mask unchanged (staging deleted, never promoted).
        np.testing.assert_array_equal(tifffile.imread(str(labels_path)), original)
        assert not list(labels_path.parent.glob("*.saving"))

    def test_promote_failure_discards_staging(self, tmp_path):
        """A promote (os.replace) failure → staging deleted, cleanup_ok True,
        original intact."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _save_curation_persist,
        )
        import tifffile
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        labels_path = labels_dir / "labels_3d.tif"
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        tifffile.imwrite(str(labels_path), original, bigtiff=True)
        kwargs, _ = self._kwargs(tmp_path, labels_path=labels_path)  # mask-only

        _real_replace = os.replace

        def failing_replace(src, dst):
            if str(dst) == str(labels_path):
                raise OSError("promote failed")
            return _real_replace(src, dst)

        with patch("os.replace", side_effect=failing_replace):
            res = _save_curation_persist(**kwargs)
        assert res["ok"] is False
        assert res["stage"] == "promote"
        assert res["cleanup_ok"] is True
        np.testing.assert_array_equal(tifffile.imread(str(labels_path)), original)
        assert not list(labels_path.parent.glob("*.saving"))

    def test_success_rotates_bak_copy(self, tmp_path):
        """On a mask+CSV success with a pre-existing original, a .bak COPY is
        left next to the promoted mask."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _save_curation_persist,
        )
        import tifffile
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        labels_path = labels_dir / "labels_3d.tif"
        original = np.zeros((4, 8, 8), dtype=np.uint32)
        original[0, 0, 0] = 3
        tifffile.imwrite(str(labels_path), original, bigtiff=True)

        csv_path = tmp_path / "feat.csv"
        df = pd.DataFrame({"label": [1]})
        kwargs, _ = self._kwargs(
            tmp_path, mask=np.full((4, 8, 8), 5, dtype=np.uint32),
            csv_df=df, csv_path=csv_path, labels_path=labels_path,
        )
        res = _save_curation_persist(**kwargs)
        assert res["ok"] is True
        bak = labels_path.with_suffix(labels_path.suffix + ".bak")
        assert bak.exists(), ".bak COPY must be rotated on success"
        # .bak holds the ORIGINAL (pre-save) data; promoted file holds the new.
        np.testing.assert_array_equal(tifffile.imread(str(bak)), original)
        np.testing.assert_array_equal(
            tifffile.imread(str(labels_path)),
            np.full((4, 8, 8), 5, dtype=np.uint32),
        )


class TestSaveCurationAsyncReentryGate:
    """_save_in_flight must gate re-entry on the async Save path: a click while
    a save is already running is ignored (no prep, no new launch)."""

    def test_in_flight_blocks_reentry(self, harness, tmp_path):
        _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        # Simulate a save already in flight.
        harness._save_in_flight = True

        prep_calls = []
        _real_prep = harness._save_curation_prepare

        def tracking_prep():
            prep_calls.append(True)
            return _real_prep()
        harness._save_curation_prepare = tracking_prep

        harness._save_curation_async()

        assert not prep_calls, (
            "_save_curation_async must early-return without running prep when a "
            "save is already in flight"
        )
        # Dirty flag untouched (no save ran).
        assert harness._curation_dirty is True


class TestSaveCurationStaleFinish:
    """A stale finish (token bumped or active patch drifted after a switch that
    slipped the lock) must NOT clobber the new patch's in-memory dirty state."""

    def test_stale_token_skips_dirty_clear(self, harness, tmp_path):
        _setup_csv(harness, tmp_path)
        # The NEW patch (after a switch) has its own dirty state.
        harness._curation_dirty = True
        harness._qc_view_patch_name = "new_patch.tif"
        # Current token is 5; the in-flight save dispatched at token 4.
        harness._save_curation_token = 5
        harness._save_in_flight = True

        result = {"ok": True, "stage": "done", "mask_only": False,
                  "warnings": []}
        # Finish for the SUPERSEDED dispatch (token 4, old patch).
        harness._on_save_curation_finished(result, 4, "old_patch.tif")

        # In-memory dirty state NOT clobbered — the new patch survives.
        assert harness._curation_dirty is True
        # In-flight flag cleared + bar hidden (finally block runs).
        assert harness._save_in_flight is False

    def test_patch_drift_skips_dirty_clear(self, harness, tmp_path):
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        # Token matches, but the active QC-view patch drifted post-dispatch.
        harness._save_curation_token = 3
        harness._qc_view_patch_name = "patchB.tif"
        harness._save_in_flight = True

        result = {"ok": True, "stage": "done", "mask_only": False,
                  "warnings": []}
        harness._on_save_curation_finished(result, 3, "patchA.tif")

        assert harness._curation_dirty is True
        assert harness._save_in_flight is False


class TestSaveCurationHandlerInvariants:
    """_on_save_curation_finished / _on_save_curation_error reproduce the legacy
    transactional branches: _curation_dirty stays TRUE on any failure, cleared
    only on full success.  _tiers_dirty is never touched by Save."""

    def test_ok_false_keeps_curation_dirty(self, harness, tmp_path):
        harness._curation_dirty = True
        harness._tiers_dirty = True
        harness._save_in_flight = True
        harness._qc_view_patch_name = "test_patch.tif"

        result = {"ok": False, "stage": "promote", "error": "promote failed",
                  "cleanup_ok": True, "mask_only": False, "warnings": []}
        harness._on_save_curation_finished(
            result, harness._save_curation_token, "test_patch.tif"
        )

        assert harness._curation_dirty is True, "failure must keep dirty"
        # Save never touches _tiers_dirty.
        assert harness._tiers_dirty is True
        assert harness._save_in_flight is False
        err = [m for m in harness._log if "[ERROR]" in m and "Staging file" in m]
        assert err, "promote failure must log the staging cleanup outcome"

    def test_ok_true_clears_curation_dirty_only(self, harness, tmp_path):
        harness._curation_dirty = True
        harness._tiers_dirty = True
        harness._save_in_flight = True
        harness._qc_view_patch_name = "test_patch.tif"

        result = {"ok": True, "stage": "done", "mask_only": False,
                  "warnings": []}
        harness._on_save_curation_finished(
            result, harness._save_curation_token, "test_patch.tif"
        )

        assert harness._curation_dirty is False, "success must clear curation dirty"
        # Save persists mask + features CSV; tier-only edits cleared elsewhere.
        assert harness._tiers_dirty is True, (
            "Save must NOT clear _tiers_dirty"
        )
        assert "test_patch.tif" in harness._edited_patches
        assert harness._save_in_flight is False

    def test_worker_error_keeps_curation_dirty(self, harness, tmp_path):
        harness._curation_dirty = True
        harness._tiers_dirty = True
        harness._save_in_flight = True

        harness._on_save_curation_error(
            "boom\ntraceback...", harness._save_curation_token, "test_patch.tif"
        )

        assert harness._curation_dirty is True
        assert harness._tiers_dirty is True
        assert harness._save_in_flight is False
        error_logs = [m for m in harness._log if "[ERROR]" in m]
        assert error_logs, "worker error must log [ERROR]"


class TestSaveCurationDirtyEpochBackstop:
    """B2: a mask edit committed DURING the async save (the dirty-epoch advances
    after dispatch) must NOT have its dirty flag cleared by the save's finish —
    the edit isn't in the snapshot being written, so the next Save must persist
    it.  A clean save (epoch unchanged) clears dirty exactly as before."""

    def test_edit_during_save_keeps_dirty(self, harness, tmp_path):
        harness._curation_dirty = True
        harness._save_in_flight = True
        harness._qc_view_patch_name = "test_patch.tif"
        # Save dispatched at epoch 0; an Accept committed during the save bumped
        # the epoch to 1 (that edit is NOT in the snapshot being written).
        harness._save_dispatch_dirty_epoch = 0
        harness._curation_dirty_epoch = 1

        result = {"ok": True, "stage": "done", "mask_only": False,
                  "warnings": []}
        harness._on_save_curation_finished(
            result, harness._save_curation_token, "test_patch.tif"
        )

        assert harness._curation_dirty is True, (
            "an edit committed during the async save must keep dirty so the "
            "next Save persists it"
        )
        # Disk WAS written for the dispatch patch + in-flight cleared.
        assert "test_patch.tif" in harness._edited_patches
        assert harness._save_in_flight is False

    def test_no_edit_during_save_clears_dirty(self, harness, tmp_path):
        harness._curation_dirty = True
        harness._save_in_flight = True
        harness._qc_view_patch_name = "test_patch.tif"
        # Epoch unchanged between dispatch and finish — a clean save.
        harness._save_dispatch_dirty_epoch = 2
        harness._curation_dirty_epoch = 2

        result = {"ok": True, "stage": "done", "mask_only": False,
                  "warnings": []}
        harness._on_save_curation_finished(
            result, harness._save_curation_token, "test_patch.tif"
        )

        assert harness._curation_dirty is False, (
            "a clean save (no edit during it) must clear dirty as before"
        )
        assert harness._save_in_flight is False


class TestOnSaveCurationClicked:
    """The Save BUTTON handler saves tiers synchronously + launches the async
    mask save; nothing happens when neither flag is dirty."""

    def test_clean_state_no_op(self, harness, tmp_path):
        harness._curation_dirty = False
        harness._tiers_dirty = False
        async_calls = []
        harness._save_curation_async = lambda: async_calls.append(True)
        harness._on_save_curation_clicked()
        assert not async_calls, "clean state must not launch the async save"

    def test_curation_dirty_launches_async(self, harness, tmp_path):
        harness._curation_dirty = True
        harness._tiers_dirty = False
        async_calls = []
        harness._save_curation_async = lambda: async_calls.append(True)
        harness._on_save_curation_clicked()
        assert async_calls, "a dirty mask must launch the async save"

    def test_tiers_dirty_saved_sync_before_async(self, harness, tmp_path):
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        harness._tiers_dirty = True
        order = []
        _real_tiers = harness._save_tiers_csv

        def tracking_tiers():
            order.append("tiers")
            _real_tiers()
        harness._save_tiers_csv = tracking_tiers
        harness._save_curation_async = lambda: order.append("async")
        harness._on_save_curation_clicked()
        assert order == ["tiers", "async"], (
            "tiers must persist synchronously BEFORE the async mask save"
        )

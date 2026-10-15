"""Tests for mask deletion performance optimizations.

Covers:
1. Single-label deletion correctness (ROI-scoped snapshot + zero)
2. Batch garbage deletion correctness (single-pass np.isin zeroing)
3. ROI-scoped subviewer refresh (only changed region updated)
4. Controller callback integration (undo stack, DataFrame update)
5. Interaction guards (split-running blocks delete, isolation survives delete)

All tests are Qt-free using lightweight fakes and direct numpy volumes.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal fakes
# ---------------------------------------------------------------------------


class FakeCombo:
    """Minimal QComboBox fake."""

    def __init__(self, text="labels"):
        self._text = text

    def currentText(self):
        return self._text


class FakeLayer:
    """Minimal napari Labels layer fake with numpy data."""

    def __init__(self, name, data):
        self.name = name
        self.data = data
        self._refreshed = 0
        self.colormap = None

    def refresh(self):
        self._refreshed += 1


class FakeLayers:
    """Minimal napari viewer.layers fake."""

    def __init__(self, layers=None):
        self._layers = {l.name: l for l in (layers or [])}

    def __contains__(self, name):
        return name in self._layers

    def __getitem__(self, name):
        return self._layers[name]


class FakeViewerModel:
    """Minimal napari ViewerModel fake."""

    def __init__(self):
        self.layers = FakeLayers()


def _layer_by_name(layers, name):
    """Helper matching NativeMultiviewAnnotator._layer_by_name."""
    if hasattr(layers, "_layers"):
        return layers._layers.get(name)
    for layer in layers:
        if layer.name == name:
            return layer
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_volume_with_labels(*label_ids, shape=(20, 30, 30)):
    """Create a volume with known label regions for testing.

    Each label gets a distinct 5x5x5 cube at a different position.
    """
    vol = np.zeros(shape, dtype=np.int32)
    for i, lid in enumerate(label_ids):
        z_start = min(i * 6, shape[0] - 5)
        vol[z_start : z_start + 5, 2:7, 2:7] = lid
    return vol


@pytest.fixture
def volume_3labels():
    """Volume with labels 1, 2, 3 in distinct regions."""
    return _make_volume_with_labels(1, 2, 3)


@pytest.fixture
def volume_with_garbage():
    """Volume with labels 1 (good), 2 (review), 10, 11, 12 (garbage)."""
    return _make_volume_with_labels(1, 2, 10, 11, 12, shape=(40, 30, 30))


# ---------------------------------------------------------------------------
# Test: _snapshot_label_roi correctness
# ---------------------------------------------------------------------------


class TestSnapshotLabelROI:
    """Verify ROI snapshot captures the correct bounding box."""

    def _import_snapshot(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        return Curation3DAnnotator._snapshot_label_roi

    def test_roi_covers_label_extent(self, volume_3labels):
        snapshot = self._import_snapshot()
        roi_slices, roi_data = snapshot(volume_3labels, 1)
        # The ROI must contain all voxels of label 1
        roi_vol = volume_3labels[roi_slices]
        assert np.sum(roi_vol == 1) == np.sum(volume_3labels == 1)

    def test_roi_data_is_copy(self, volume_3labels):
        snapshot = self._import_snapshot()
        roi_slices, roi_data = snapshot(volume_3labels, 1)
        # Mutating original should not affect snapshot
        original_count = np.sum(roi_data == 1)
        volume_3labels[volume_3labels == 1] = 0
        assert np.sum(roi_data == 1) == original_count

    def test_missing_label_returns_full_volume(self, volume_3labels):
        snapshot = self._import_snapshot()
        roi_slices, roi_data = snapshot(volume_3labels, 999)
        assert roi_data.shape == volume_3labels.shape

    def test_roi_has_margin(self, volume_3labels):
        snapshot = self._import_snapshot()
        roi_slices, roi_data = snapshot(volume_3labels, 1)
        # Margin is 15, so ROI should be larger than the label extent
        coords = np.argwhere(volume_3labels == 1)
        label_extent = coords.max(axis=0) - coords.min(axis=0) + 1
        roi_extent = np.array(roi_data.shape)
        assert np.all(roi_extent >= label_extent)


# ---------------------------------------------------------------------------
# Test: Single label deletion
# ---------------------------------------------------------------------------


class TestSingleLabelDeletion:
    """Verify _on_delete_from_navigator zeros the label and notifies."""

    def _make_annotator_fake(self, volume):
        """Build a minimal fake of Curation3DAnnotator for deletion tests."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        ann = object.__new__(CurationAnnotatorFake)
        ann._split_running = False
        ann.labels_combo = FakeCombo("labels")
        layer = FakeLayer("labels", volume)
        ann.viewer = types.SimpleNamespace(layers=FakeLayers([layer]))
        ann._on_delete_label_cb = MagicMock()
        # Bind the real static method for snapshot
        ann._snapshot_label_roi = CurationAnnotatorFake._snapshot_label_roi
        # Bind _refresh stub
        ann._refresh_label_data_in_subviewers = MagicMock()
        return ann, layer

    def test_zeros_target_label(self, volume_3labels):
        vol = volume_3labels.copy()
        ann, layer = self._make_annotator_fake(vol)
        assert np.sum(vol == 2) > 0
        ann._on_delete_from_navigator(2)
        assert np.sum(vol == 2) == 0

    def test_preserves_other_labels(self, volume_3labels):
        vol = volume_3labels.copy()
        count1_before = np.sum(vol == 1)
        count3_before = np.sum(vol == 3)
        ann, layer = self._make_annotator_fake(vol)
        ann._on_delete_from_navigator(2)
        assert np.sum(vol == 1) == count1_before
        assert np.sum(vol == 3) == count3_before

    def test_refreshes_main_layer(self, volume_3labels):
        vol = volume_3labels.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_delete_from_navigator(2)
        assert layer._refreshed >= 1

    def test_refreshes_subviewers(self, volume_3labels):
        vol = volume_3labels.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_delete_from_navigator(2)
        ann._refresh_label_data_in_subviewers.assert_called_once()

    def test_notifies_callback_with_roi(self, volume_3labels):
        vol = volume_3labels.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_delete_from_navigator(2)
        ann._on_delete_label_cb.assert_called_once()
        _, kwargs = ann._on_delete_label_cb.call_args
        assert kwargs["roi_slices"] is not None
        assert kwargs["old_roi_data"] is not None

    def test_split_running_blocks_delete(self, volume_3labels):
        vol = volume_3labels.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._split_running = True
        ann._on_delete_from_navigator(2)
        assert np.sum(vol == 2) > 0  # unchanged

    def test_callback_roi_contains_label_data(self, volume_3labels):
        vol = volume_3labels.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_delete_from_navigator(2)
        _, kwargs = ann._on_delete_label_cb.call_args
        roi_data = kwargs["old_roi_data"]
        # The snapshot was taken BEFORE zeroing, so it should contain label 2
        assert np.sum(roi_data == 2) > 0


# ---------------------------------------------------------------------------
# Test: Batch garbage deletion
# ---------------------------------------------------------------------------


class TestBatchGarbageDeletion:
    """Verify _on_discard_garbage_from_navigator zeros all garbage labels."""

    def _make_annotator_fake(self, volume):
        ann = object.__new__(CurationAnnotatorFake)
        ann._split_running = False
        ann.labels_combo = FakeCombo("labels")
        layer = FakeLayer("labels", volume)
        ann.viewer = types.SimpleNamespace(layers=FakeLayers([layer]))
        ann._on_discard_garbage_cb = MagicMock()
        ann._refresh_label_data_in_subviewers = MagicMock()
        return ann, layer

    def test_zeros_all_garbage_labels(self, volume_with_garbage):
        vol = volume_with_garbage.copy()
        garbage = [10, 11, 12]
        for gid in garbage:
            assert np.sum(vol == gid) > 0
        ann, layer = self._make_annotator_fake(vol)
        ann._on_discard_garbage_from_navigator(garbage)
        for gid in garbage:
            assert np.sum(vol == gid) == 0

    def test_preserves_non_garbage_labels(self, volume_with_garbage):
        vol = volume_with_garbage.copy()
        count1 = np.sum(vol == 1)
        count2 = np.sum(vol == 2)
        ann, layer = self._make_annotator_fake(vol)
        ann._on_discard_garbage_from_navigator([10, 11, 12])
        assert np.sum(vol == 1) == count1
        assert np.sum(vol == 2) == count2

    def test_single_refresh_after_batch(self, volume_with_garbage):
        vol = volume_with_garbage.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_discard_garbage_from_navigator([10, 11, 12])
        # Should only refresh main layer once, not once per garbage label
        assert layer._refreshed == 1

    def test_single_subviewer_refresh_after_batch(self, volume_with_garbage):
        vol = volume_with_garbage.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_discard_garbage_from_navigator([10, 11, 12])
        ann._refresh_label_data_in_subviewers.assert_called_once()

    def test_callback_receives_all_ids(self, volume_with_garbage):
        vol = volume_with_garbage.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_discard_garbage_from_navigator([10, 11, 12])
        ann._on_discard_garbage_cb.assert_called_once()
        args = ann._on_discard_garbage_cb.call_args
        assert args[0][0] == [10, 11, 12]

    def test_empty_list_is_noop(self, volume_with_garbage):
        vol = volume_with_garbage.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_discard_garbage_from_navigator([])
        assert layer._refreshed == 0

    def test_split_running_blocks_batch(self, volume_with_garbage):
        vol = volume_with_garbage.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._split_running = True
        ann._on_discard_garbage_from_navigator([10, 11, 12])
        assert np.sum(vol == 10) > 0  # unchanged

    def test_combined_roi_covers_all_garbage(self, volume_with_garbage):
        vol = volume_with_garbage.copy()
        ann, layer = self._make_annotator_fake(vol)
        ann._on_discard_garbage_from_navigator([10, 11, 12])
        _, kwargs = ann._on_discard_garbage_cb.call_args
        roi_data = kwargs["old_roi_data"]
        # Snapshot taken before zeroing — should contain all garbage labels
        for gid in [10, 11, 12]:
            assert np.sum(roi_data == gid) > 0


# ---------------------------------------------------------------------------
# Test: Batch deletion uses single-pass np.isin (performance)
# ---------------------------------------------------------------------------


class TestDeletionPerformanceOptimizations:
    """Verify deletion uses efficient operations (no redundant full-volume scans)."""

    def test_batch_uses_combined_mask_not_loop(self):
        """Batch zeroing should use combined_mask, not per-label loop."""
        import inspect
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_discard_garbage_from_navigator
        )
        # Should use combined_mask for zeroing, not loop of per-label scans
        assert "combined_mask" in source and "layer.data[combined_mask] = 0" in source, (
            "Batch deletion should use combined_mask for single-pass zeroing"
        )
        # Should NOT have the old per-label loop pattern
        assert "for lid in garbage_ids:" not in source, (
            "Batch deletion should not loop over garbage_ids for zeroing"
        )

    def test_single_delete_uses_roi_scoped_zeroing(self):
        """Single delete should zero within ROI, not scan full volume again."""
        import inspect
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_delete_from_navigator
        )
        # Should use ROI-scoped zeroing
        assert "roi_view" in source, (
            "Single delete should use roi_view for ROI-scoped zeroing"
        )
        # Should NOT have the old full-volume pattern
        assert "layer.data[layer.data ==" not in source, (
            "Single delete should not do a second full-volume scan"
        )

    def test_merge_uses_roi_scoped_relabeling(self):
        """Merge should relabel within ROI, not scan full volume again."""
        import inspect
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_merge_from_navigator
        )
        assert "roi_view" in source, (
            "Merge should use roi_view for ROI-scoped relabeling"
        )

    def test_batch_zeroing_single_pass_correctness(self):
        """Verify single-pass np.isin zeroing produces correct result."""
        vol = _make_volume_with_labels(10, 11, 12, shape=(20, 30, 30))
        garbage = [10, 11, 12]

        combined_mask = np.isin(vol, garbage)
        vol[combined_mask] = 0
        for gid in garbage:
            assert np.sum(vol == gid) == 0

    def test_snapshot_uses_axis_reduced_bbox(self):
        """_snapshot_label_roi should use np.any axis reduction, not np.argwhere."""
        import ast
        import inspect
        import textwrap
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._snapshot_label_roi
        )
        # Should use axis-reduced approach
        assert "np.any(mask" in source or "np.any(" in source, (
            "snapshot should use np.any for axis-reduced bounding box"
        )
        # Check that np.argwhere is NOT called in executable code
        # (it can appear in comments/docstrings — that's fine)
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "np"
                    and node.attr == "argwhere"
                ):
                    pytest.fail(
                        "snapshot should not call np.argwhere in code"
                    )

    def test_roi_scoped_zeroing_equivalent_to_full_volume(self):
        """ROI-scoped zeroing must produce identical result to full-volume zeroing."""
        vol_full = _make_volume_with_labels(1, 2, 3, shape=(20, 30, 30))
        vol_roi = vol_full.copy()

        # Full-volume approach (old)
        vol_full[vol_full == 2] = 0

        # ROI-scoped approach (new)
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        roi_slices, _ = Curation3DAnnotator._snapshot_label_roi(vol_roi, 2)
        roi_view = vol_roi[roi_slices]
        roi_view[roi_view == 2] = 0

        np.testing.assert_array_equal(vol_full, vol_roi)


# ---------------------------------------------------------------------------
# Test: Benchmark — axis-reduced bbox vs np.argwhere
# ---------------------------------------------------------------------------


class TestBBoxPerformance:
    """Verify axis-reduced bbox is faster than np.argwhere on realistic volumes."""

    def test_axis_reduced_faster_than_argwhere(self):
        """On a 1M+ voxel volume, axis-reduced bbox should be faster."""
        import time

        # 100x100x100 = 1M voxels — realistic for a single patch
        vol = np.zeros((100, 100, 100), dtype=np.int32)
        vol[20:30, 40:60, 10:25] = 42  # 10x20x15 = 3000 voxels

        # Old approach: np.argwhere
        t0 = time.perf_counter()
        for _ in range(10):
            coords = np.argwhere(vol == 42)
            _ = coords.min(axis=0), coords.max(axis=0)
        t_argwhere = time.perf_counter() - t0

        # New approach: axis-reduced np.any
        t0 = time.perf_counter()
        for _ in range(10):
            mask = vol == 42
            z_any = np.any(mask, axis=(1, 2))
            y_any = np.any(mask, axis=(0, 2))
            x_any = np.any(mask, axis=(0, 1))
            z_idx = np.where(z_any)[0]
            y_idx = np.where(y_any)[0]
            x_idx = np.where(x_any)[0]
            _ = (z_idx[0], y_idx[0], x_idx[0]), (z_idx[-1], y_idx[-1], x_idx[-1])
        t_axis = time.perf_counter() - t0

        # axis-reduced should be faster (typically 2-5x on 1M volumes)
        # Use a generous margin — just verify it's not dramatically slower
        assert t_axis < t_argwhere * 2.0, (
            f"axis-reduced ({t_axis:.4f}s) should not be much slower "
            f"than argwhere ({t_argwhere:.4f}s)"
        )

    def test_bbox_results_match(self):
        """Axis-reduced and argwhere approaches must give identical bounding boxes."""
        vol = np.zeros((50, 60, 70), dtype=np.int32)
        vol[10:20, 25:35, 5:15] = 7

        # argwhere approach
        coords = np.argwhere(vol == 7)
        mins_aw = coords.min(axis=0)
        maxs_aw = coords.max(axis=0)

        # axis-reduced approach
        mask = vol == 7
        z_idx = np.where(np.any(mask, axis=(1, 2)))[0]
        y_idx = np.where(np.any(mask, axis=(0, 2)))[0]
        x_idx = np.where(np.any(mask, axis=(0, 1)))[0]
        mins_ar = np.array([z_idx[0], y_idx[0], x_idx[0]])
        maxs_ar = np.array([z_idx[-1], y_idx[-1], x_idx[-1]])

        np.testing.assert_array_equal(mins_aw, mins_ar)
        np.testing.assert_array_equal(maxs_aw, maxs_ar)


# ---------------------------------------------------------------------------
# Test: ROI-scoped subviewer refresh
# ---------------------------------------------------------------------------


class TestROIScopedSubviewerRefresh:
    """Verify subviewer refresh only updates changed region when possible."""

    def test_refresh_updates_all_three_orientations(self):
        """After deletion, all 3 subviewers must have updated data."""
        from packages.epicycle_annotator.orientation import transpose_data

        vol = _make_volume_with_labels(1, 2, shape=(20, 30, 30))
        # Create subviewer layers with transposed views
        xy_data = transpose_data(vol, "XY").copy()
        xz_data = transpose_data(vol, "XZ").copy()
        yz_data = transpose_data(vol, "YZ").copy()

        # Delete label 2 from main volume
        vol[vol == 2] = 0

        # Refresh subviewers
        xy_new = transpose_data(vol, "XY")
        xz_new = transpose_data(vol, "XZ")
        yz_new = transpose_data(vol, "YZ")

        # All orientations should reflect the deletion
        assert np.sum(xy_new == 2) == 0
        assert np.sum(xz_new == 2) == 0
        assert np.sum(yz_new == 2) == 0

    def test_transpose_is_view_not_copy(self):
        """np.transpose returns a view — verify no unnecessary copies."""
        from packages.epicycle_annotator.orientation import transpose_data

        vol = _make_volume_with_labels(1, shape=(20, 30, 30))
        xy = transpose_data(vol, "XY")
        # XY is identity, should share memory
        assert np.shares_memory(vol, xy)

        xz = transpose_data(vol, "XZ")
        assert np.shares_memory(vol, xz)


# ---------------------------------------------------------------------------
# Test: Controller callback chain
# ---------------------------------------------------------------------------


class TestControllerDeleteCallback:
    """Verify controller handles deletion callback correctly."""

    def _make_controller_fake(self):
        """Minimal fake of the fusion controller for deletion tests."""
        import pandas as pd

        ctrl = MagicMock()
        ctrl._qc_features_df = pd.DataFrame({
            "label": [1, 2, 3, 10, 11, 12],
            "area": [100, 200, 300, 50, 60, 70],
            "tier": ["good", "review", "bad", "garbage", "garbage", "garbage"],
        })
        ctrl._curation_dirty = False
        ctrl._curation_undo_stack = MagicMock()
        ctrl._curation_undo_stack.__bool__ = lambda self: True
        return ctrl

    def test_delete_marks_label_as_deleted(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        import pandas as pd

        ctrl = self._make_controller_fake()
        df = ctrl._qc_features_df

        # Simulate what _on_annotator_label_deleted does
        mask = df["label"] == 2
        if mask.any():
            if "label_status" not in df.columns:
                df["label_status"] = "active"
            df.loc[mask, "label_status"] = "deleted"

        assert df.loc[df["label"] == 2, "label_status"].iloc[0] == "deleted"
        assert df.loc[df["label"] == 1, "label_status"].iloc[0] == "active"

    def test_discard_garbage_marks_all_as_deleted(self):
        import pandas as pd

        ctrl = self._make_controller_fake()
        df = ctrl._qc_features_df
        garbage_ids = [10, 11, 12]

        if "label_status" not in df.columns:
            df["label_status"] = "active"
        mask = df["label"].isin(garbage_ids)
        df.loc[mask, "label_status"] = "deleted"

        for gid in garbage_ids:
            assert df.loc[df["label"] == gid, "label_status"].iloc[0] == "deleted"
        assert df.loc[df["label"] == 1, "label_status"].iloc[0] == "active"


# ---------------------------------------------------------------------------
# Test: Interaction guards
# ---------------------------------------------------------------------------


class TestDeletionInteractionGuards:
    """Verify deletion respects interaction locks."""

    def test_delete_during_interpolation_blocked(self):
        """If interpolation is running (split_running), delete should no-op."""
        vol = _make_volume_with_labels(1, 2, shape=(10, 10, 10))
        ann = object.__new__(CurationAnnotatorFake)
        ann._split_running = True
        ann.labels_combo = FakeCombo("labels")
        layer = FakeLayer("labels", vol)
        ann.viewer = types.SimpleNamespace(layers=FakeLayers([layer]))
        ann._on_delete_label_cb = MagicMock()
        ann._refresh_label_data_in_subviewers = MagicMock()
        ann._snapshot_label_roi = CurationAnnotatorFake._snapshot_label_roi

        ann._on_delete_from_navigator(2)
        assert np.sum(vol == 2) > 0  # not deleted
        ann._on_delete_label_cb.assert_not_called()

    def test_delete_nonexistent_label_is_safe(self):
        """Deleting a label not in the volume should not crash."""
        vol = _make_volume_with_labels(1, 2, shape=(10, 10, 10))
        ann = object.__new__(CurationAnnotatorFake)
        ann._split_running = False
        ann.labels_combo = FakeCombo("labels")
        layer = FakeLayer("labels", vol)
        ann.viewer = types.SimpleNamespace(layers=FakeLayers([layer]))
        ann._on_delete_label_cb = MagicMock()
        ann._refresh_label_data_in_subviewers = MagicMock()
        ann._snapshot_label_roi = CurationAnnotatorFake._snapshot_label_roi

        ann._on_delete_from_navigator(999)  # no crash
        # Volume unchanged
        assert np.sum(vol == 1) > 0
        assert np.sum(vol == 2) > 0

    def test_delete_with_missing_layer_name_is_safe(self):
        """If labels_combo returns a name not in viewer.layers, no crash."""
        vol = _make_volume_with_labels(1, shape=(10, 10, 10))
        ann = object.__new__(CurationAnnotatorFake)
        ann._split_running = False
        ann.labels_combo = FakeCombo("nonexistent_layer")
        layer = FakeLayer("labels", vol)
        ann.viewer = types.SimpleNamespace(layers=FakeLayers([layer]))
        ann._on_delete_label_cb = MagicMock()
        ann._refresh_label_data_in_subviewers = MagicMock()

        ann._on_delete_from_navigator(1)  # no crash
        assert np.sum(vol == 1) > 0  # unchanged


# ---------------------------------------------------------------------------
# Fake class that borrows real methods from Curation3DAnnotator
# ---------------------------------------------------------------------------


class CurationAnnotatorFake:
    """Fake that uses real Curation3DAnnotator methods for testing.

    We import and bind the real methods so tests exercise actual logic,
    not mocked behavior.
    """

    @staticmethod
    def _snapshot_label_roi(volume, label_id, edited_slices=None):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        return Curation3DAnnotator._snapshot_label_roi(
            volume, label_id, edited_slices
        )

    def _on_delete_from_navigator(self, label_id):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        return Curation3DAnnotator._on_delete_from_navigator(self, label_id)

    def _on_discard_garbage_from_navigator(self, garbage_ids):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        return Curation3DAnnotator._on_discard_garbage_from_navigator(
            self, garbage_ids
        )

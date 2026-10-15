"""End-to-end interpolation tests — multi-orientation, ROI snapshot, neighbor protection.

Phase K11d — comprehensive curation test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.interpolation import (
    interpolate_label_all_orientations,
    interpolate_label_in_volume,
    postprocess_label,
)


# ---------------------------------------------------------------------------
# Tests — interpolate_label_all_orientations
# ---------------------------------------------------------------------------


class TestInterpolateLabelAllOrientations:
    def test_xy_only(self):
        """Interpolation with annotations only in XY orientation."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        label_id = 1
        # Paint circles on slices 2 and 6
        volume[2, 8:12, 8:12] = label_id
        volume[6, 8:12, 8:12] = label_id

        edited = {
            "XY": {label_id: [2, 6]},
            "XZ": {},
            "YZ": {},
        }

        total = interpolate_label_all_orientations(volume, label_id, edited)
        assert total > 0
        # Intermediate slices (3, 4, 5) should now have label_id
        for z in [3, 4, 5]:
            assert (volume[z] == label_id).any()

    def test_xz_only(self):
        """Interpolation with annotations only in XZ orientation."""
        volume = np.zeros((20, 10, 20), dtype=np.int32)
        label_id = 2
        volume[:, 2, 8:12] = label_id  # Y-slice 2
        volume[:, 6, 8:12] = label_id  # Y-slice 6

        edited = {
            "XY": {},
            "XZ": {label_id: [2, 6]},
            "YZ": {},
        }

        total = interpolate_label_all_orientations(volume, label_id, edited)
        assert total > 0
        for y in [3, 4, 5]:
            assert (volume[:, y, :] == label_id).any()

    def test_yz_only(self):
        """Interpolation with annotations only in YZ orientation."""
        volume = np.zeros((20, 20, 10), dtype=np.int32)
        label_id = 3
        volume[:, 8:12, 2] = label_id  # X-slice 2
        volume[:, 8:12, 6] = label_id  # X-slice 6

        edited = {
            "XY": {},
            "XZ": {},
            "YZ": {label_id: [2, 6]},
        }

        total = interpolate_label_all_orientations(volume, label_id, edited)
        assert total > 0
        for x in [3, 4, 5]:
            assert (volume[:, :, x] == label_id).any()

    def test_multi_orientation(self):
        """Interpolation across both XY and XZ orientations."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        label_id = 1
        volume[2, 8:12, 8:12] = label_id
        volume[8, 8:12, 8:12] = label_id
        volume[:, 3, 8:12] = label_id
        volume[:, 7, 8:12] = label_id

        edited = {
            "XY": {label_id: [2, 8]},
            "XZ": {label_id: [3, 7]},
            "YZ": {},
        }

        total = interpolate_label_all_orientations(volume, label_id, edited)
        assert total > 0

    def test_single_slice_per_orient_skips(self):
        """With only 1 edited slice per orientation, no interpolation occurs."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        label_id = 1
        volume[5, 8:12, 8:12] = label_id

        edited = {
            "XY": {label_id: [5]},
            "XZ": {},
            "YZ": {},
        }

        total = interpolate_label_all_orientations(volume, label_id, edited)
        assert total == 0

    def test_empty_edited_slices(self):
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        total = interpolate_label_all_orientations(volume, 1, {"XY": {}, "XZ": {}, "YZ": {}})
        assert total == 0

    def test_numpy_int_key_handled(self):
        """edited_slices keys may be numpy integers (from UI)."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        label_id = np.int64(5)
        volume[2, 8:12, 8:12] = int(label_id)
        volume[6, 8:12, 8:12] = int(label_id)

        edited = {
            "XY": {np.int64(5): [2, 6]},
            "XZ": {},
            "YZ": {},
        }

        total = interpolate_label_all_orientations(volume, int(label_id), edited)
        assert total > 0


# ---------------------------------------------------------------------------
# Tests — neighbor protection
# ---------------------------------------------------------------------------


class TestNeighborProtection:
    def test_interpolation_does_not_overwrite_neighbors(self):
        """Interpolation should only write to background (0) or own label."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        label_a = 1
        label_b = 2

        # Label A on slices 0 and 4
        volume[0, 5:15, 5:15] = label_a
        volume[4, 5:15, 5:15] = label_a

        # Label B occupies the same region on slice 2
        volume[2, 5:15, 5:15] = label_b

        total = interpolate_label_in_volume(volume, label_a, [0, 4], "XY")

        # Slice 2 should still have label_b where it was
        assert (volume[2, 5:15, 5:15] == label_b).all()

    def test_postprocess_does_not_overwrite_neighbors(self):
        """Postprocessing (fill + closing) should respect neighbor labels."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        label_a = 1
        label_b = 2

        # Label A: small blob
        volume[3:7, 8:12, 8:12] = label_a
        # Poke a hole inside label A
        volume[5, 10, 10] = 0

        # Label B: adjacent to label A
        volume[3:7, 12:16, 8:12] = label_b

        postprocess_label(volume, label_a)

        # The hole should be filled with label_a
        assert volume[5, 10, 10] == label_a
        # Label B should be untouched
        assert (volume[3:7, 12:16, 8:12] == label_b).all()


# ---------------------------------------------------------------------------
# Tests — postprocessing
# ---------------------------------------------------------------------------


class TestPostprocessLabel:
    def test_fills_internal_holes(self):
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        label_id = 1
        volume[3:7, 5:15, 5:15] = label_id
        # Create a hole
        volume[5, 10, 10] = 0

        postprocess_label(volume, label_id)
        assert volume[5, 10, 10] == label_id

    def test_empty_label_is_noop(self):
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        postprocess_label(volume, 99)  # Should not raise

    def test_single_voxel_label(self):
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        volume[5, 10, 10] = 7
        postprocess_label(volume, 7)  # Should not raise
        assert volume[5, 10, 10] == 7


# ---------------------------------------------------------------------------
# Tests — ROI snapshot (from Curation3DAnnotator._snapshot_label_roi)
# ---------------------------------------------------------------------------


class TestROISnapshot:
    """Test the bbox-based ROI snapshot/restore cycle for interpolation preview.

    The snapshot covers the label's full bounding box + generous margin so that
    postprocessing expansion is fully captured on reject.
    """

    @staticmethod
    def _snapshot_label_roi(volume, label_id, edited_slices=None):
        """Mirror of Curation3DAnnotator._snapshot_label_roi (bbox-based)."""
        coords = np.argwhere(volume == label_id)
        if coords.size == 0:
            roi = (
                slice(0, volume.shape[0]),
                slice(0, volume.shape[1]),
                slice(0, volume.shape[2]),
            )
            return roi, volume[roi].copy()

        shape = volume.shape
        margin = 15

        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        z_lo = max(0, int(mins[0]) - margin)
        z_hi = min(shape[0], int(maxs[0]) + margin + 1)
        y_lo = max(0, int(mins[1]) - margin)
        y_hi = min(shape[1], int(maxs[1]) + margin + 1)
        x_lo = max(0, int(mins[2]) - margin)
        x_hi = min(shape[2], int(maxs[2]) + margin + 1)

        roi = (slice(z_lo, z_hi), slice(y_lo, y_hi), slice(x_lo, x_hi))
        return roi, volume[roi].copy()

    def test_snapshot_covers_full_label_bbox(self):
        """Snapshot ROI should cover the label's full bounding box, not just edited slices."""
        volume = np.zeros((50, 30, 30), dtype=np.int32)
        # Label spans Z=5 to Z=40
        volume[5, 15, 15] = 1
        volume[40, 15, 15] = 1

        roi, snapshot = self._snapshot_label_roi(volume, 1)

        # ROI should cover Z=5-40 + margin 15 → Z=[0, 50)
        assert roi[0].start == 0
        assert roi[0].stop == 50  # clamped to volume bounds

    def test_snapshot_captures_label_extent_not_edited_slices(self):
        """Even if only 2 slices are edited, snapshot covers the label's full extent."""
        volume = np.zeros((100, 30, 30), dtype=np.int32)
        # Label exists from Z=0 to Z=99
        volume[0, 15, 15] = 1
        volume[99, 15, 15] = 1
        # User only edited Z=40 and Z=60, but label spans full Z range
        roi, snapshot = self._snapshot_label_roi(volume, 1)

        # Snapshot must cover the full Z range, not just Z=40-60
        assert roi[0].start == 0
        assert roi[0].stop == 100

    def test_snapshot_is_independent_copy(self):
        volume = np.zeros((20, 30, 30), dtype=np.int32)
        volume[5, 10, 10] = 1

        roi, snapshot = self._snapshot_label_roi(volume, 1)

        # Modify volume after snapshot
        volume[5, 10, 10] = 99
        assert snapshot[snapshot > 0].max() == 1  # still has original value

    def test_snapshot_restore_cycle(self):
        """Preview → snapshot → modify → reject (restore) should revert volume."""
        volume = np.zeros((20, 30, 30), dtype=np.int32)
        volume[5, 10:20, 10:20] = 1
        volume[15, 10:20, 10:20] = 1
        original = volume.copy()

        roi, snapshot = self._snapshot_label_roi(volume, 1)

        # Simulate interpolation modifying the volume
        volume[roi] = 999

        # Restore
        volume[roi] = snapshot
        np.testing.assert_array_equal(volume, original)

    def test_snapshot_restore_with_postprocessing_expansion(self):
        """Snapshot must cover postprocessing expansion so reject fully restores."""
        volume = np.zeros((30, 30, 30), dtype=np.int32)
        label_id = 1
        # Label occupies Z=5 to Z=20
        volume[5, 10:20, 10:20] = label_id
        volume[20, 10:20, 10:20] = label_id
        original = volume.copy()

        roi, snapshot = self._snapshot_label_roi(volume, label_id)

        # Simulate interpolation + postprocessing
        interpolate_label_all_orientations(
            volume, label_id,
            {"XY": {label_id: [5, 20]}, "XZ": {}, "YZ": {}},
        )

        # Volume was modified
        assert not np.array_equal(volume, original)

        # Restore and verify full restoration
        volume[roi] = snapshot
        np.testing.assert_array_equal(volume, original)

    def test_roi_margin_clamped_to_bounds(self):
        """ROI margin should not exceed volume boundaries."""
        volume = np.zeros((10, 10, 10), dtype=np.int32)
        volume[0, 5, 5] = 1
        volume[2, 5, 5] = 1
        roi, snapshot = self._snapshot_label_roi(volume, 1)

        assert roi[0].start >= 0
        assert roi[0].stop <= 10
        assert roi[1].start >= 0
        assert roi[1].stop <= 10
        assert roi[2].start >= 0
        assert roi[2].stop <= 10

    def test_label_not_in_volume_falls_back_to_full(self):
        """If label doesn't exist in volume, snapshot covers full volume."""
        volume = np.zeros((10, 10, 10), dtype=np.int32)
        roi, snapshot = self._snapshot_label_roi(volume, 99)
        assert snapshot.shape == volume.shape

    def test_3d_bbox_covers_all_axes(self):
        """Snapshot ROI should scope all three axes from the label's extent."""
        volume = np.zeros((40, 40, 40), dtype=np.int32)
        # Label at specific 3D positions
        volume[5, 10, 20] = 1
        volume[30, 25, 35] = 1

        roi, snapshot = self._snapshot_label_roi(volume, 1)

        # Z: 5-30 + margin 15 → [0, 40)
        assert roi[0].start == 0
        assert roi[0].stop == 40  # clamped
        # Y: 10-25 + margin 15 → [0, 40)
        assert roi[1].start == 0
        assert roi[1].stop == 40
        # X: 20-35 + margin 15 → [5, 40)
        assert roi[2].start == max(0, 20 - 15)
        assert roi[2].stop == 40


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestInterpolationEdgeCases:
    def test_adjacent_slices_no_gap(self):
        """Consecutive annotated slices (gap=0) should produce 0 interpolations."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        volume[3, 8:12, 8:12] = 1
        volume[4, 8:12, 8:12] = 1

        total = interpolate_label_in_volume(volume, 1, [3, 4], "XY")
        assert total == 0

    def test_duplicate_slice_indices(self):
        """Duplicate indices in edited_slices should be deduplicated."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        volume[2, 8:12, 8:12] = 1
        volume[6, 8:12, 8:12] = 1

        total = interpolate_label_in_volume(volume, 1, [2, 2, 6, 6], "XY")
        assert total > 0

    def test_wrong_label_id_no_interpolation(self):
        """Requesting interpolation for a label not in the volume at edited slices."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        volume[2, 8:12, 8:12] = 1  # only label 1 exists
        volume[6, 8:12, 8:12] = 1

        total = interpolate_label_in_volume(volume, 99, [2, 6], "XY")
        assert total == 0

    def test_unsorted_slice_indices(self):
        """Slice indices can be unsorted — should still work."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        volume[2, 8:12, 8:12] = 1
        volume[6, 8:12, 8:12] = 1

        total = interpolate_label_in_volume(volume, 1, [6, 2], "XY")
        assert total > 0

    def test_volume_boundary_interpolation(self):
        """Interpolation at volume edges (slice 0 and max) should work."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        volume[0, 8:12, 8:12] = 1
        volume[9, 8:12, 8:12] = 1

        total = interpolate_label_in_volume(volume, 1, [0, 9], "XY")
        assert total > 0
        # All intermediate slices should have some voxels
        for z in range(1, 9):
            assert (volume[z] == 1).any()


# ---------------------------------------------------------------------------
# Tests — Preview / Accept / Reject state machine
# ---------------------------------------------------------------------------

# Lightweight fake that mirrors Curation3DAnnotator's interpolation state
# without requiring Qt/napari.


class _FakeSyncManager:
    """Minimal fake for sync_manager used by interpolation state tests."""

    def __init__(self):
        self.shared_labels: dict[str, np.ndarray] = {}
        self._edited: dict[str, dict] = {}

    def get_edited_slices(self, layer_name: str) -> dict:
        return self._edited.setdefault(layer_name, {"XY": {}, "XZ": {}, "YZ": {}})

    def set_edited_slices(self, layer_name: str, label_id: int, orient: str, indices: list):
        edited = self.get_edited_slices(layer_name)
        edited[orient][label_id] = list(indices)


class _FakeAnnotator:
    """Minimal fake mirroring Curation3DAnnotator interpolation state."""

    def __init__(self, volume: np.ndarray, label_name: str = "labels"):
        self.sync_manager = _FakeSyncManager()
        self.sync_manager.shared_labels[label_name] = volume
        self._active_labels_name = label_name
        self._interp_snapshot = None
        self._interp_roi = None
        self._interp_label_id = None
        self._interp_running = False
        self._status_text = ""
        self._accept_enabled = False
        self._reject_enabled = False

    @staticmethod
    def _snapshot_label_roi(volume, label_id, edited_slices=None):
        """Mirror of the bbox-based snapshot logic."""
        coords = np.argwhere(volume == label_id)
        if coords.size == 0:
            roi = (
                slice(0, volume.shape[0]),
                slice(0, volume.shape[1]),
                slice(0, volume.shape[2]),
            )
            return roi, volume[roi].copy()

        shape = volume.shape
        margin = 15
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        z_lo = max(0, int(mins[0]) - margin)
        z_hi = min(shape[0], int(maxs[0]) + margin + 1)
        y_lo = max(0, int(mins[1]) - margin)
        y_hi = min(shape[1], int(maxs[1]) + margin + 1)
        x_lo = max(0, int(mins[2]) - margin)
        x_hi = min(shape[2], int(maxs[2]) + margin + 1)
        roi = (slice(z_lo, z_hi), slice(y_lo, y_hi), slice(x_lo, x_hi))
        return roi, volume[roi].copy()

    def run_preview(self, selected_label: int):
        """Simulate _run_interpolation() flow."""
        if self._interp_running:
            return
        self._interp_running = True
        try:
            shared_data = self.sync_manager.shared_labels.get(self._active_labels_name)
            if shared_data is None:
                self._status_text = "No shared label volume found."
                return
            edited = self.sync_manager.get_edited_slices(self._active_labels_name)
            has_slices = False
            for orient_dict in edited.values():
                for lid, slices in orient_dict.items():
                    if int(lid) == selected_label and len(slices) >= 2:
                        has_slices = True
                        break
                if has_slices:
                    break
            if not has_slices:
                self._status_text = f"Label {selected_label}: need >= 2 annotated slices."
                return
            self._interp_roi, self._interp_snapshot = self._snapshot_label_roi(
                shared_data, selected_label, edited
            )
            self._interp_label_id = selected_label
            total = interpolate_label_all_orientations(shared_data, selected_label, edited)
            self._status_text = f"Label {selected_label}: {total} slice(s) interpolated."
            self._accept_enabled = True
            self._reject_enabled = True
        finally:
            self._interp_running = False

    def accept(self):
        """Simulate _accept_interpolation() flow."""
        if self._interp_label_id is None:
            self._status_text = "No active interpolation to accept."
            return
        self._interp_snapshot = None
        self._interp_roi = None
        label_id = self._interp_label_id
        self._interp_label_id = None
        labels_name = self._active_labels_name
        if labels_name:
            edited = self.sync_manager.get_edited_slices(labels_name)
            for orient_dict in edited.values():
                orient_dict.pop(label_id, None)
                for key in list(orient_dict.keys()):
                    if int(key) == label_id:
                        orient_dict.pop(key, None)
        self._accept_enabled = False
        self._reject_enabled = False
        self._status_text = f"Accepted label {label_id}."

    def reject(self):
        """Simulate _reject_interpolation() flow."""
        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is None:
            self._status_text = "ERROR: Label volume not found. Cannot reject."
            self._interp_snapshot = None
            self._interp_roi = None
            self._interp_label_id = None
            self._accept_enabled = False
            self._reject_enabled = False
            return
        if self._interp_snapshot is not None and self._interp_roi is not None:
            shared_data[self._interp_roi] = self._interp_snapshot
        if labels_name and self._interp_label_id is not None:
            edited = self.sync_manager.get_edited_slices(labels_name)
            for orient_dict in edited.values():
                orient_dict.pop(self._interp_label_id, None)
                for key in list(orient_dict.keys()):
                    if int(key) == self._interp_label_id:
                        orient_dict.pop(key, None)
        self._interp_snapshot = None
        self._interp_roi = None
        self._interp_label_id = None
        self._accept_enabled = False
        self._reject_enabled = False
        self._status_text = "Interpolation rejected. Volume restored."


class TestPreviewAcceptRejectStateMachine:
    """Test the full Preview → Accept/Reject state machine."""

    @staticmethod
    def _make_annotator_with_label(label_id=1, z_slices=(2, 8)):
        """Create a fake annotator with a label painted on two slices."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        for z in z_slices:
            volume[z, 8:12, 8:12] = label_id
        ann = _FakeAnnotator(volume)
        ann.sync_manager.set_edited_slices("labels", label_id, "XY", list(z_slices))
        return ann, volume

    def test_preview_accept_clears_edited_slices(self):
        """After Accept, edited_slices for that label should be cleared."""
        ann, volume = self._make_annotator_with_label()
        ann.run_preview(1)
        assert ann._accept_enabled

        ann.accept()

        edited = ann.sync_manager.get_edited_slices("labels")
        assert 1 not in edited["XY"]

    def test_preview_reject_clears_edited_slices(self):
        """After Reject, edited_slices for that label should be cleared (Bug #1)."""
        ann, volume = self._make_annotator_with_label()
        ann.run_preview(1)

        ann.reject()

        edited = ann.sync_manager.get_edited_slices("labels")
        assert 1 not in edited["XY"]

    def test_preview_reject_restores_volume(self):
        """Reject should fully restore the volume to pre-interpolation state."""
        ann, volume = self._make_annotator_with_label()
        original = volume.copy()

        ann.run_preview(1)
        # Interpolation should have modified intermediate slices
        assert not np.array_equal(volume, original)

        ann.reject()
        np.testing.assert_array_equal(volume, original)

    def test_preview_accept_keeps_interpolation(self):
        """Accept should keep the interpolated data in the volume."""
        ann, volume = self._make_annotator_with_label()
        original = volume.copy()

        ann.run_preview(1)
        # Intermediate slices should have label data
        for z in [3, 4, 5, 6, 7]:
            assert (volume[z] == 1).any()

        ann.accept()
        # Data should still be there
        for z in [3, 4, 5, 6, 7]:
            assert (volume[z] == 1).any()

    def test_accept_without_preview_is_noop(self):
        """Accept without prior Preview should not crash (Bug #2 guard)."""
        ann, volume = self._make_annotator_with_label()
        ann.accept()
        assert "No active interpolation" in ann._status_text

    def test_reject_with_missing_volume_shows_error(self):
        """Reject when volume is missing should show error, not silent success (Bug #3)."""
        ann, volume = self._make_annotator_with_label()
        ann.run_preview(1)

        # Remove the volume
        ann.sync_manager.shared_labels.clear()

        ann.reject()
        assert "ERROR" in ann._status_text

    def test_concurrent_preview_guard(self):
        """Second Preview while first is running should be rejected (Bug #5)."""
        ann, volume = self._make_annotator_with_label()
        # Simulate that interpolation is running
        ann._interp_running = True

        ann.run_preview(1)

        # Should not have set up snapshot (was rejected)
        assert ann._interp_snapshot is None

    def test_reject_then_preview_again_needs_new_edits(self):
        """After reject, Preview should require fresh edits (edited_slices cleared)."""
        ann, volume = self._make_annotator_with_label()
        ann.run_preview(1)
        ann.reject()

        # Edited slices should be cleared
        edited = ann.sync_manager.get_edited_slices("labels")
        assert 1 not in edited["XY"]

        # Preview again without new edits should fail
        ann.run_preview(1)
        assert "need >= 2" in ann._status_text

    def test_preview_preserves_user_paintings_on_reject(self):
        """User's hand-painted corrections should survive Reject."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        label_id = 1

        # Simulate existing label from reconstruction
        volume[2, 8:12, 8:12] = label_id
        volume[8, 8:12, 8:12] = label_id

        # Simulate user painting corrections (expanded region on slice 2)
        volume[2, 6:14, 6:14] = label_id
        volume[8, 6:14, 6:14] = label_id
        painted_state = volume.copy()

        ann = _FakeAnnotator(volume)
        ann.sync_manager.set_edited_slices("labels", label_id, "XY", [2, 8])

        ann.run_preview(label_id)
        # Interpolation modifies intermediate slices
        assert (volume[5] == label_id).any()

        ann.reject()
        # Volume should be back to the painted state (user corrections preserved)
        np.testing.assert_array_equal(volume, painted_state)

    def test_accept_clears_state_flags(self):
        """After Accept, all interpolation state should be cleared."""
        ann, volume = self._make_annotator_with_label()
        ann.run_preview(1)
        ann.accept()

        assert ann._interp_snapshot is None
        assert ann._interp_roi is None
        assert ann._interp_label_id is None
        assert not ann._accept_enabled
        assert not ann._reject_enabled

    def test_reject_clears_state_flags(self):
        """After Reject, all interpolation state should be cleared."""
        ann, volume = self._make_annotator_with_label()
        ann.run_preview(1)
        ann.reject()

        assert ann._interp_snapshot is None
        assert ann._interp_roi is None
        assert ann._interp_label_id is None
        assert not ann._accept_enabled
        assert not ann._reject_enabled

    def test_refinement_workflow_existing_label(self):
        """Full refinement workflow: existing label → paint corrections → interpolate → accept."""
        volume = np.zeros((30, 30, 30), dtype=np.int32)
        label_id = 5

        # Existing reconstructed label with a gap in the middle
        volume[2, 10:20, 10:20] = label_id
        volume[5, 10:20, 10:20] = label_id
        # Gap at Z=3,4 — no label data there
        volume[10, 10:20, 10:20] = label_id

        ann = _FakeAnnotator(volume)
        # User paints corrections on Z=2 and Z=10 to refine
        ann.sync_manager.set_edited_slices("labels", label_id, "XY", [2, 10])

        ann.run_preview(label_id)

        # Gap should now be filled
        for z in [3, 4, 5, 6, 7, 8, 9]:
            assert (volume[z] == label_id).any(), f"Z={z} should have label after interpolation"

        ann.accept()
        # Interpolated data persists
        for z in [3, 4, 5, 6, 7, 8, 9]:
            assert (volume[z] == label_id).any()


# ---------------------------------------------------------------------------
# Tests — anchor-aware interpolation (existing label + user corrections)
# ---------------------------------------------------------------------------


class TestAnchorAwareInterpolation:
    """Verify interpolation uses existing label voxels as implicit anchors.

    When a label already exists in the volume (e.g., from reconstruction),
    user corrections should blend with the existing shape. The SDF
    interpolation should use ALL slices containing the label as anchors,
    not just user-edited slices.
    """

    def test_existing_label_preserves_shape(self):
        """User draws 2 corrections on an existing label. Intermediate
        slices should be filled without destroying existing data."""
        volume = np.zeros((20, 30, 30), dtype=np.int32)
        label_id = 3
        # Existing reconstruction: full label on slices 5-15
        for z in range(5, 16):
            volume[z, 10:20, 10:20] = label_id

        # User draws corrections on slices 8 and 12 (slightly different shape)
        volume[8, 9:21, 9:21] = label_id  # slightly larger
        volume[12, 9:21, 9:21] = label_id  # slightly larger

        edited = {"XY": {label_id: [8, 12]}, "XZ": {}, "YZ": {}}
        total = interpolate_label_all_orientations(
            volume, label_id, edited, post_process=False
        )
        # All slices 5-15 should still have the label
        for z in range(5, 16):
            assert np.any(volume[z] == label_id), (
                f"Slice {z} should still have label after interpolation"
            )

    def test_de_novo_still_works(self):
        """Pure de novo label (no existing data) still interpolates correctly."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        label_id = 1
        # User draws on slices 2 and 7 only (no prior)
        volume[2, 8:12, 8:12] = label_id
        volume[7, 8:12, 8:12] = label_id

        edited = {"XY": {label_id: [2, 7]}, "XZ": {}, "YZ": {}}
        total = interpolate_label_all_orientations(
            volume, label_id, edited, post_process=False
        )
        assert total > 0
        for z in range(3, 7):
            assert np.any(volume[z] == label_id), (
                f"Slice {z} should be filled by de novo interpolation"
            )

    def test_no_crosshatch_artifact(self):
        """Multi-orientation interpolation on existing label should not
        create cross-hatch artifacts (the original bug)."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        label_id = 2
        # Existing spherical-ish label
        for z in range(5, 15):
            for y in range(5, 15):
                for x in range(5, 15):
                    if (z - 10)**2 + (y - 10)**2 + (x - 10)**2 < 20:
                        volume[z, y, x] = label_id

        # User draws corrections in XY and XZ
        volume[7, 6:14, 6:14] = label_id  # XY correction
        volume[12, 6:14, 6:14] = label_id  # XY correction
        volume[6:14, 8, 6:14] = label_id  # XZ correction
        volume[6:14, 11, 6:14] = label_id  # XZ correction

        edited = {
            "XY": {label_id: [7, 12]},
            "XZ": {label_id: [8, 11]},
            "YZ": {},
        }
        total = interpolate_label_all_orientations(
            volume, label_id, edited, post_process=True
        )
        # The result should be a smooth filled volume, not striped.
        # Check that the center region is solidly filled.
        center_region = volume[8:12, 8:12, 8:12]
        fill_ratio = np.sum(center_region == label_id) / center_region.size
        assert fill_ratio > 0.8, (
            f"Center should be mostly filled (ratio={fill_ratio:.2f}), "
            f"not have cross-hatch gaps"
        )

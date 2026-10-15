"""Tests for manual-draw tracking — fresh slices without segmentation.

Bug fixed: in `_on_annotator_draw_complete`, missing `sample_record` caused
silent bail.  Result: user paints a slice without segmenting first → work
is never tracked, never persisted as a training pair.  Data loss.

Fix shape:
- `_touch_sample` accepts new optional `origin` kwarg (forward-compat for
  cellpose / sam / stardist / omnipose / watershed / scribble / manual etc.)
- New `_process_draw_complete(slice_label, view, idx)` helper holds the
  decision logic from `_on_annotator_draw_complete` so it's testable
  without Qt-level dims reads.
- For a missing record: creates one with status="curating", origin="manual".
- C1 fix: ALSO registers input_stack in segmentation_manager.data_inputs
  for fresh manual draws so the debounced persist can write a real
  training pair (without this, persist silently no-ops).
- C3 fix: skip phantom records when the work_labels volume at that slice
  has no painted content (e.g. erase-only stroke on empty layer).
- Existing predicted→curating transition unchanged.
"""

from __future__ import annotations

import numpy as np

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


# ---------------------------------------------------------------------------
# Fake viewer + layers for C1/C3 integration tests
# ---------------------------------------------------------------------------


class _FakeImageLayer:
    """Stand-in for an Image layer with patch-view naming + 3D data."""

    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data
        self.visible = True


# _get_patch_view_image_layers checks layer.__class__.__name__ == "Image".
# Set the class name post-definition so we don't shadow napari.layers.Image.
_FakeImageLayer.__name__ = "Image"


class _FakeLabelsLayer:
    """Stand-in for a Labels layer with patch + view naming + 3D volume."""

    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data
        self.visible = True


_FakeLabelsLayer.__name__ = "Labels"


class _FakeLayerList:
    """Iterable + indexable layer collection matching napari's surface."""

    def __init__(self, layers):
        self._layers = list(layers)
        self._by_name = {lyr.name: lyr for lyr in layers}

    def __iter__(self):
        return iter(self._layers)

    def __contains__(self, name):
        return name in self._by_name

    def __getitem__(self, name):
        return self._by_name[name]


class _FakeViewer:
    def __init__(self, layers):
        self.layers = _FakeLayerList(layers)


class _DummySegMgr:
    """Mirrors test_dataflow_fixes.py::DummySegmentationManager."""

    def __init__(self):
        self.data_names = {}
        self.data_inputs = {}
        self.data_masks = {}

    def get_input_stack(self, slice_label):
        return self.data_inputs.get(slice_label)

    def get_masks(self, slice_label):
        return self.data_masks.get(slice_label)


def _make_minimal_ctrl(sample_records=None):
    """Bare-minimum controller for testing _touch_sample + _process_draw_complete.

    Avoids HITLPersistenceService / TrainingManager / SegmentationManager
    setup — we only need state fields + a no-op _schedule_curating_persist
    + a no-op _refresh_sample_queue stub.
    """
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.loaded_patch_name = "patch_test"
    ctrl.selected_view = "XY"
    ctrl.sample_records = dict(sample_records or {})
    ctrl.schedule_calls: list = []
    ctrl.refresh_calls: int = 0

    def _stub_schedule(slice_label, view, s_idx):
        ctrl.schedule_calls.append((slice_label, view, s_idx))

    def _stub_refresh():
        ctrl.refresh_calls += 1

    ctrl._schedule_curating_persist = _stub_schedule
    ctrl._refresh_sample_queue = _stub_refresh
    # 2026-05-27 edit-states: the accepted/staged demote branch + committed
    # protection branch call these — stub so the minimal ctrl supports them.
    ctrl.edit_state_refresh_calls = 0

    def _stub_edit_state_refresh():
        ctrl.edit_state_refresh_calls += 1

    ctrl._refresh_annotator_edit_state_visuals = _stub_edit_state_refresh
    ctrl._append_ui_log = lambda *a, **k: None
    return ctrl


# ---------------------------------------------------------------------------
# _touch_sample with new `origin` field
# ---------------------------------------------------------------------------


def test_touch_sample_accepts_origin_kwarg():
    """New `origin` kwarg gets persisted on the record."""
    ctrl = _make_minimal_ctrl()
    ctrl._touch_sample("patch_test_XY_42", status="curating", view="XY", origin="manual")

    rec = ctrl.sample_records["patch_test_XY_42"]
    assert rec["origin"] == "manual"
    assert rec["status"] == "curating"
    assert rec["view"] == "XY"


def test_touch_sample_origin_preserved_across_updates():
    """Subsequent _touch_sample calls without `origin` keep the existing value."""
    ctrl = _make_minimal_ctrl()
    ctrl._touch_sample("patch_test_XY_42", status="curating", view="XY", origin="manual")
    # Later transition (e.g., user clicks Accept) without specifying origin
    ctrl._touch_sample("patch_test_XY_42", status="accepted", view="XY")

    rec = ctrl.sample_records["patch_test_XY_42"]
    assert rec["origin"] == "manual", "origin must persist across status updates"
    assert rec["status"] == "accepted"


def test_touch_sample_defaults_origin_to_none_when_unspecified():
    """Brand-new record without explicit origin gets None (unknown).

    Allows older code paths + future migration to not require explicit values.
    """
    ctrl = _make_minimal_ctrl()
    ctrl._touch_sample("patch_test_XY_42", status="predicted", view="XY")

    rec = ctrl.sample_records["patch_test_XY_42"]
    # Either None or unset — both signal "no provenance recorded".
    assert rec.get("origin") in (None, "")


# ---------------------------------------------------------------------------
# _process_draw_complete decision logic
# ---------------------------------------------------------------------------


def test_fresh_manual_draw_creates_curating_record_with_origin_manual():
    """REGRESSION (the bug being fixed): a draw stroke on a slice with no
    existing sample_record must create one as curating + origin=manual,
    refresh the queue, and schedule persist."""
    ctrl = _make_minimal_ctrl()

    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)

    rec = ctrl.sample_records.get("patch_test_XY_42")
    assert rec is not None, "Fresh manual draw must create a sample_record"
    assert rec["status"] == "curating"
    assert rec["origin"] == "manual"
    assert rec["view"] == "XY"
    assert ctrl.refresh_calls == 1, "Queue must refresh so user sees the new entry"
    assert ctrl.schedule_calls == [("patch_test_XY_42", "XY", 42)]


def test_predicted_slice_transitions_to_curating_on_draw():
    """Existing behavior preserved: a draw on a predicted slice transitions
    status to curating + schedules persist.  Origin stays "cellpose"
    (originally written by segment path)."""
    ctrl = _make_minimal_ctrl(
        sample_records={
            "patch_test_XY_42": {
                "sample_id": "patch_test_XY_42",
                "patch_id": "patch_test",
                "view": "XY",
                "slice_index": 42,
                "bucket": "lateral",
                "status": "predicted",
                "mask_ref": "patch_test_XY_42",
                "updated_at": "old",
                "round_tag": "round_00",
                "origin": "cellpose",
            }
        }
    )

    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)

    rec = ctrl.sample_records["patch_test_XY_42"]
    assert rec["status"] == "curating", "Must transition predicted -> curating"
    assert rec["origin"] == "cellpose", "Origin from segment path must be preserved"
    assert ctrl.refresh_calls == 1
    assert ctrl.schedule_calls == [("patch_test_XY_42", "XY", 42)]


def test_curating_slice_only_schedules_persist_no_status_change():
    """Existing behavior: drawing on an already-curating slice doesn't
    re-touch status — just debounces the persist."""
    ctrl = _make_minimal_ctrl(
        sample_records={
            "patch_test_XY_42": {
                "sample_id": "patch_test_XY_42",
                "patch_id": "patch_test",
                "view": "XY",
                "slice_index": 42,
                "bucket": "lateral",
                "status": "curating",
                "mask_ref": "patch_test_XY_42",
                "updated_at": "old",
                "round_tag": "round_00",
                "origin": "manual",
            }
        }
    )

    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)

    rec = ctrl.sample_records["patch_test_XY_42"]
    assert rec["status"] == "curating"
    assert rec["updated_at"] == "old", "No _touch_sample call → updated_at unchanged"
    assert ctrl.refresh_calls == 0, "No status transition → no queue refresh"
    assert ctrl.schedule_calls == [("patch_test_XY_42", "XY", 42)]


def test_accepted_slice_demotes_to_curating_on_draw():
    """2026-05-27 edit-states: BEHAVIOR CHANGE (was the no-op
    'test_accepted_slice_does_not_reschedule_persist').

    Drawing on an accepted slice now DEMOTES it to curating + schedules a
    persist + refreshes the annotator soft-lock visuals.  Pre-fix this was
    a silent no-op — the subviewer layer mutated visually but nothing
    persisted, so training used the OLD mask (the confirmed divergence).
    The annotator's soft visual lock discourages accidental edits; a
    deliberate paint lands here and reopens the slice for editing.
    """
    ctrl = _make_minimal_ctrl(
        sample_records={
            "patch_test_XY_42": {
                "sample_id": "patch_test_XY_42",
                "patch_id": "patch_test",
                "view": "XY",
                "slice_index": 42,
                "bucket": "lateral",
                "status": "accepted",
                "mask_ref": "patch_test_XY_42",
                "updated_at": "old",
                "round_tag": "round_00",
                "origin": "cellpose",
            }
        }
    )

    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)

    rec = ctrl.sample_records["patch_test_XY_42"]
    assert rec["status"] == "curating", "accepted slice must demote to curating on edit"
    assert rec["origin"] == "cellpose", "origin provenance must survive the demote"
    assert ctrl.refresh_calls == 1, "queue must refresh so user sees the demote"
    assert ctrl.schedule_calls == [("patch_test_XY_42", "XY", 42)], (
        "persist must be scheduled so the edit lands on disk"
    )
    assert ctrl.edit_state_refresh_calls == 1, "annotator soft-lock visuals must refresh"


def test_staged_slice_demotes_to_curating_on_draw():
    """staged behaves like accepted — demotes + persists on edit."""
    ctrl = _make_minimal_ctrl(
        sample_records={
            "patch_test_XY_42": {
                "sample_id": "patch_test_XY_42", "patch_id": "patch_test",
                "view": "XY", "slice_index": 42, "bucket": "lateral",
                "status": "staged", "mask_ref": "patch_test_XY_42",
                "updated_at": "old", "round_tag": "round_00", "origin": "manual",
            }
        }
    )
    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)
    rec = ctrl.sample_records["patch_test_XY_42"]
    assert rec["status"] == "curating"
    assert rec["origin"] == "manual"
    assert ctrl.schedule_calls == [("patch_test_XY_42", "XY", 42)]


def test_committed_slice_protected_on_draw():
    """committed is PROTECTED — drawing is a no-op (no demote, no persist).
    User must use the main-widget Revert to reopen a committed slice."""
    ctrl = _make_minimal_ctrl(
        sample_records={
            "patch_test_XY_42": {
                "sample_id": "patch_test_XY_42", "patch_id": "patch_test",
                "view": "XY", "slice_index": 42, "bucket": "lateral",
                "status": "committed", "mask_ref": "patch_test_XY_42",
                "updated_at": "old", "round_tag": "round_00", "origin": "cellpose",
            }
        }
    )
    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)
    rec = ctrl.sample_records["patch_test_XY_42"]
    assert rec["status"] == "committed", "committed must NOT regress on draw"
    assert ctrl.schedule_calls == [], "no persist for protected committed slices"


def test_fresh_axial_draw_xz_creates_record_with_correct_bucket():
    """Manual draw on axial XZ subviewer creates record with bucket=axial."""
    ctrl = _make_minimal_ctrl()

    ctrl._process_draw_complete("patch_test_XZ_100", "XZ", 100)

    rec = ctrl.sample_records.get("patch_test_XZ_100")
    assert rec is not None
    assert rec["status"] == "curating"
    assert rec["origin"] == "manual"
    assert rec["view"] == "XZ"
    assert rec["bucket"] == "axial"


def test_fresh_axial_draw_yz_creates_record_with_correct_bucket():
    """Symmetric: manual draw on YZ also bucket=axial."""
    ctrl = _make_minimal_ctrl()

    ctrl._process_draw_complete("patch_test_YZ_200", "YZ", 200)

    rec = ctrl.sample_records.get("patch_test_YZ_200")
    assert rec is not None
    assert rec["view"] == "YZ"
    assert rec["bucket"] == "axial"


# ---------------------------------------------------------------------------
# C1 + C3 integration tests — exercise the real input_stack registration
# and empty-stroke skip paths (not just the decision logic).
# ---------------------------------------------------------------------------


def _make_integration_ctrl(image_data, labels_data, view="XY"):
    """Controller with a fake viewer containing one image + one work_labels
    layer for the given view.  Used to exercise the C1 + C3 code paths
    that read from real layers.
    """
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.loaded_patch_name = "patch_test"
    ctrl.selected_view = view
    ctrl.sample_records = {}
    ctrl.segmentation_manager = _DummySegMgr()
    ctrl.schedule_calls: list = []
    ctrl.refresh_calls: int = 0

    image_name = f"patch_test_ch0_{view}_view"
    if view == "XY":
        labels_name = "patch_test_work_labels_lateral_3d"
    elif view == "XZ":
        labels_name = "patch_test_work_labels_axial_xz_3d"
    else:
        labels_name = "patch_test_work_labels_axial_yz_3d"

    image_layer = _FakeImageLayer(image_name, image_data)
    labels_layer = _FakeLabelsLayer(labels_name, labels_data)
    ctrl.viewer = _FakeViewer([image_layer, labels_layer])

    def _stub_schedule(slice_label, view, s_idx):
        ctrl.schedule_calls.append((slice_label, view, s_idx))

    def _stub_refresh():
        ctrl.refresh_calls += 1

    ctrl._schedule_curating_persist = _stub_schedule
    ctrl._refresh_sample_queue = _stub_refresh
    # 2026-05-27 edit-states: the accepted/staged demote branch + committed
    # protection branch call these — stub so the minimal ctrl supports them.
    ctrl.edit_state_refresh_calls = 0

    def _stub_edit_state_refresh():
        ctrl.edit_state_refresh_calls += 1

    ctrl._refresh_annotator_edit_state_visuals = _stub_edit_state_refresh
    ctrl._append_ui_log = lambda *a, **k: None
    return ctrl


def test_c1_fresh_manual_draw_registers_input_stack_in_data_inputs():
    """C1 regression: fresh manual draw populates segmentation_manager.data_inputs
    so the debounced persist can later write a real training pair to disk.

    Without this, _persist_curated_training_pair silently returns {} because
    get_input_stack(slice_label) is None, and pending.pop() unconditionally
    drops the entry → no img.tif / mask.tif files written → work lost on
    napari close before Accept.
    """
    image = np.arange(141 * 512 * 512, dtype=np.uint16).reshape(141, 512, 512)
    labels = np.zeros((141, 512, 512), dtype=np.int32)
    labels[42, 100:200, 100:200] = 5  # painted content at slice 42
    ctrl = _make_integration_ctrl(image, labels, view="XY")

    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)

    # Record was created
    assert "patch_test_XY_42" in ctrl.sample_records
    rec = ctrl.sample_records["patch_test_XY_42"]
    assert rec["status"] == "curating"
    assert rec["origin"] == "manual"

    # AND input_stack got registered — the C1 fix.
    assert ctrl.segmentation_manager.get_input_stack("patch_test_XY_42") is not None
    input_stack = ctrl.segmentation_manager.data_inputs["patch_test_XY_42"]
    # Should be (1 channel, H, W) — 1 image layer, sliced at idx=42 along axis 0.
    assert input_stack.shape == (1, 512, 512), f"Expected (1, 512, 512), got {input_stack.shape}"
    # And data_names registered too (mirrors segment-path bookkeeping)
    assert ctrl.segmentation_manager.data_names.get("patch_test_XY_42") is True


def test_c1_fresh_manual_draw_input_stack_for_axial_xz():
    """C1: axial XZ view input extraction (slice axis = 1, not 0)."""
    # XZ image layer shape: (Y, Z, X) for a typical patch.
    # We use (512, 141, 512) so axis 1 is the slicing axis.
    image = np.arange(512 * 141 * 512, dtype=np.uint16).reshape(512, 141, 512)
    labels = np.zeros((512, 141, 512), dtype=np.int32)
    labels[:, 100, 50:150] = 7  # painted content at axis-1 idx 100
    ctrl = _make_integration_ctrl(image, labels, view="XZ")

    ctrl._process_draw_complete("patch_test_XZ_100", "XZ", 100)

    input_stack = ctrl.segmentation_manager.data_inputs.get("patch_test_XZ_100")
    assert input_stack is not None
    # XZ slicing extracts shape (Y, X) → after stack: (C, Y, X) = (1, 512, 512)
    assert input_stack.shape == (1, 512, 512)


def test_c3_empty_stroke_on_fresh_slice_skips_record_creation():
    """C3: erase-only or no-paint stroke on a fresh empty slice does NOT
    create a phantom sample_record.  Avoids cluttering the box list with
    empty masks that wouldn't survive Accept anyway.
    """
    image = np.zeros((141, 512, 512), dtype=np.uint16)
    labels = np.zeros((141, 512, 512), dtype=np.int32)  # entirely empty
    ctrl = _make_integration_ctrl(image, labels, view="XY")

    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)

    # No record should be created for an empty slice
    assert "patch_test_XY_42" not in ctrl.sample_records
    # No refresh, no schedule either
    assert ctrl.refresh_calls == 0
    assert ctrl.schedule_calls == []


def test_c3_painted_stroke_on_fresh_slice_creates_record():
    """C3 positive case: a stroke that ACTUALLY painted content creates
    the record normally (mirror of test_c1_fresh_manual_draw_registers_-
    input_stack_in_data_inputs but explicit about the C3 path)."""
    image = np.zeros((141, 512, 512), dtype=np.uint16)
    labels = np.zeros((141, 512, 512), dtype=np.int32)
    labels[42, 250, 250] = 1  # one painted pixel — enough to count
    ctrl = _make_integration_ctrl(image, labels, view="XY")

    ctrl._process_draw_complete("patch_test_XY_42", "XY", 42)

    assert "patch_test_XY_42" in ctrl.sample_records
    assert ctrl.refresh_calls == 1
    assert ctrl.schedule_calls == [("patch_test_XY_42", "XY", 42)]

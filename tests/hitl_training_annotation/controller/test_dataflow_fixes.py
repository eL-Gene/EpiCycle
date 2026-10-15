"""Tests for Phase J8 dataflow fixes.

Covers:
- Two-step accept flow (accept -> stage)
- Native accept callback registration
- Cumulative training with prior round pairs
- From-scratch training mode
- Session reload from persisted pairs
- Review tab population
"""

import json
import numpy as np
import tifffile
from pathlib import Path

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)
from napari_epicycle.hitl_training_annotation.controller.training_manager import TrainingManager
from napari_epicycle.io_registry import HITLPersistenceService


class _Item:
    def __init__(self, text):
        self._text = text
        self._foreground = None
        self._data = {}

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text

    def setForeground(self, brush):
        self._foreground = brush

    def setData(self, role, value):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)

    def setIcon(self, icon):
        pass


class _ListWidget:
    def __init__(self, items=None):
        self._items = [_Item(x) for x in (items or [])]
        self._selected = []

    def count(self):
        return len(self._items)

    def item(self, i):
        return self._items[i]

    def addItem(self, text):
        self._items.append(_Item(text))

    def clear(self):
        self._items = []

    def selectedItems(self):
        return self._selected


class _ProgressBar:
    def __init__(self):
        self.value = 0

    def setRange(self, _a, _b):
        return None

    def setValue(self, value):
        self.value = value


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _Label:
    def __init__(self):
        self.value = ""

    def setText(self, text):
        self.value = text


class _Button:
    def __init__(self):
        self._enabled = True

    def setEnabled(self, v):
        self._enabled = v

    def isEnabled(self):
        return self._enabled


class _TabWidget:
    def __init__(self):
        self._tabs = {}

    def setTabText(self, index, text):
        self._tabs[index] = text


class DummySegmentationManager:
    def __init__(self):
        self.data_names = {}
        self.data_inputs = {}
        self.data_masks = {}
        self.data_flows = {}
        self.data_cellprobs = {}

    def set(self, slice_label, inp, mask):
        self.data_names[slice_label] = True
        self.data_inputs[slice_label] = inp
        self.data_masks[slice_label] = mask

    def get_input_stack(self, slice_label):
        return self.data_inputs.get(slice_label)

    def get_masks(self, slice_label):
        return self.data_masks.get(slice_label)

    def save_interactive_mask(self, slice_label, mask):
        self.data_masks[slice_label] = mask


def _make_ctrl(tmp_path, *, round_num=0, sample_records=None):
    """Create a minimal controller instance for testing."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.training_manager = TrainingManager()
    ctrl.segmentation_manager = DummySegmentationManager()
    ctrl.working_dir = str(tmp_path)
    ctrl.current_round = round_num
    ctrl.current_config = {"model_name": "epi", "model_type": "cyto3", "epochs": 100}
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.round_backend_info = {}
    ctrl.loaded_patch_name = "patch01"
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl.axial_plane = "XZ"
    ctrl.sample_records = dict(sample_records or {})
    ctrl.training_mode = "cumulative"
    ctrl.active_curation_checkpoints = {"lateral": "", "axial": ""}
    ctrl.curation_model_source_mode = "auto"
    ctrl.imported_curation_model_path = ""
    ctrl.curation_pretrained_model_type = "cyto3"
    ctrl._use_sync_training = True

    class UI:
        pass

    ui = UI()
    ui.data_to_add_list = _ListWidget()
    ui.segment_slice_data_list = _ListWidget()
    ui.training_progress_bar = _ProgressBar()
    ui.training_log_output = _LogOutput()
    ui.running_training_loss_plot = _Label()
    ui.update_and_train_button = _Button()
    # Unified slice list — all aliases point to the same widget.
    ui.accepted_queue_list = ui.segment_slice_data_list
    ui.staged_queue_list = ui.segment_slice_data_list
    ui.sample_queue_list = ui.segment_slice_data_list
    ui.review_tabs = _TabWidget()
    ctrl.ui = ui
    return ctrl


# --- Issue 3: Two-step accept flow ---


def test_accept_sets_accepted_status_not_staged():
    """SAM-path accept should set status to 'accepted', not 'staged'."""
    ctrl = _make_ctrl(Path("/tmp/test_accept"))
    slice_label = "patch01_XY_5"
    ctrl._touch_sample(slice_label, status="predicted", view="XY")

    # Simulate what _accept_annotation does after merging mask:
    ctrl._touch_sample(slice_label, status="accepted", view="XY")

    record = ctrl.sample_records[slice_label]
    assert record["status"] == "accepted"
    assert record["bucket"] == "lateral"


def test_stage_transitions_accepted_to_staged(tmp_path):
    """Stage action should move 'accepted' samples to 'staged'."""
    ctrl = _make_ctrl(tmp_path)
    sl1 = "patch01_XY_5"
    sl2 = "patch01_XY_8"

    # Seed segmentation manager so persist can work.
    ctrl.segmentation_manager.set(sl1, np.random.rand(2, 8, 8).astype(np.float32), np.ones((8, 8), dtype=np.uint16))
    ctrl.segmentation_manager.set(sl2, np.random.rand(2, 8, 8).astype(np.float32), np.ones((8, 8), dtype=np.uint16))

    ctrl._touch_sample(sl1, status="accepted", view="XY")
    ctrl._touch_sample(sl2, status="accepted", view="XY")

    ctrl._stage_accepted_in_current_view()

    assert ctrl.sample_records[sl1]["status"] == "staged"
    assert ctrl.sample_records[sl2]["status"] == "staged"

    # Staged list should now have 2 items.
    assert ctrl.ui.data_to_add_list.count() == 2


def test_training_rejects_empty_staged_list(tmp_path):
    """Training should not proceed without explicitly staged slices."""
    ctrl = _make_ctrl(tmp_path)
    # Add a predicted slice but don't stage it.
    sl = "patch01_XY_3"
    ctrl.segmentation_manager.set(sl, np.random.rand(2, 8, 8).astype(np.float32), np.ones((8, 8), dtype=np.uint16))
    ctrl._touch_sample(sl, status="predicted", view="XY")
    ctrl.ui.segment_slice_data_list.addItem(sl)

    round_before = ctrl.current_round
    ctrl._train_model()

    assert ctrl.current_round == round_before  # Should not advance.


# --- Issue 1: Native accept callback ---


def test_native_accept_registers_in_segmentation_manager(tmp_path):
    """Native accept callback should populate segmentation_manager and set accepted."""
    ctrl = _make_ctrl(tmp_path)

    # Create a mock viewer with layers.
    volume_shape = (10, 16, 16)
    label_data = np.zeros(volume_shape, dtype=np.uint16)
    label_data[3, 5:10, 5:10] = 1  # Some labels at slice 3.

    img_data = np.random.rand(*volume_shape).astype(np.float32)

    class MockLayer:
        def __init__(self, name, data, cls_name="Image"):
            self.name = name
            self.data = data
            self._cls = cls_name

        @property
        def __class__(self):
            class _C:
                pass
            c = _C()
            c.__name__ = self._cls
            return c

    class MockLayers:
        def __init__(self, layers):
            self._layers = layers

        def __iter__(self):
            return iter(self._layers)

        def __contains__(self, name):
            return any(l.name == name for l in self._layers)

    class MockViewer:
        pass

    viewer = MockViewer()
    viewer.layers = MockLayers([
        MockLayer("patch01_ch0_XY_view", img_data, "Image"),
        MockLayer("patch01_work_labels_lateral_3d", label_data, "Labels"),
    ])

    ctrl.viewer = viewer

    result = ctrl._accept_native_annotation("XY", slice_index=3)
    assert result["ok"] is True
    assert "patch01_XY_3" in ctrl.segmentation_manager.data_masks
    assert ctrl.sample_records["patch01_XY_3"]["status"] == "accepted"


def test_native_accept_rejects_empty_mask(tmp_path):
    """Native accept should fail if mask slice is all zeros."""
    ctrl = _make_ctrl(tmp_path)

    volume_shape = (10, 16, 16)
    label_data = np.zeros(volume_shape, dtype=np.uint16)  # All zeros.
    img_data = np.random.rand(*volume_shape).astype(np.float32)

    class MockLayer:
        def __init__(self, name, data, cls_name="Image"):
            self.name = name
            self.data = data

        @property
        def __class__(self):
            class _C:
                pass
            c = _C()
            c.__name__ = cls_name
            return c

    class MockLayers:
        def __init__(self, layers):
            self._layers = layers

        def __iter__(self):
            return iter(self._layers)

        def __contains__(self, name):
            return any(l.name == name for l in self._layers)

    class MockViewer:
        pass

    viewer = MockViewer()
    viewer.layers = MockLayers([
        MockLayer("patch01_ch0_XY_view", img_data, "Image"),
        MockLayer("patch01_work_labels_lateral_3d", label_data, "Labels"),
    ])
    ctrl.viewer = viewer

    result = ctrl._accept_native_annotation("XY", slice_index=3)
    assert result["ok"] is False
    assert "Empty mask" in result["message"]


# --- Issue 2: Cross-round accumulation ---


def _write_pair(directory, stack_id, view, slice_idx, shape=(2, 8, 8)):
    """Helper to write a training pair to disk."""
    directory.mkdir(parents=True, exist_ok=True)
    fname_base = f"{stack_id}_{view}_z{slice_idx:04d}"
    img_path = directory / f"{fname_base}_img.tif"
    mask_path = directory / f"{fname_base}_mask.tif"
    tifffile.imwrite(str(img_path), np.random.rand(*shape).astype(np.float32))
    tifffile.imwrite(str(mask_path), np.ones(shape[1:], dtype=np.uint16))
    return str(img_path), str(mask_path)


def test_cumulative_training_collects_prior_rounds(tmp_path):
    """With prior round data on disk, _collect_prior_round_pairs returns only
    pairs matching the current training bucket (lateral → spatial)."""
    ctrl = _make_ctrl(tmp_path, round_num=2)

    # Create round_00 and round_01 data on disk (in accepted_pairs/).
    _write_pair(tmp_path / "training_data" / "round_00" / "accepted_pairs" / "spatial", "s1", "XY", 5)
    _write_pair(tmp_path / "training_data" / "round_00" / "accepted_pairs" / "axial", "s1", "XZ", 3)
    _write_pair(tmp_path / "training_data" / "round_01" / "accepted_pairs" / "spatial", "s2", "XY", 7)
    # B-lite (2026-09-07): a prior-round pair only trains when its slice
    # record says the user accepted it.
    for label in ("s1_XY_z0005", "s1_XZ_z0003", "s2_XY_z0007"):
        ctrl._touch_sample(label, status="accepted")

    pairs = ctrl._collect_prior_round_pairs()
    # Only spatial pairs (lateral bucket), axial filtered out.
    assert len(pairs) == 2
    assert all(p["view_group"] == "spatial" for p in pairs)


def test_cumulative_training_includes_current_accepted_pairs(tmp_path):
    """Prior round accepted_pairs should also be scanned (persisted pairs live there)."""
    ctrl = _make_ctrl(tmp_path, round_num=1)

    # Prior round data in accepted_pairs (written by _persist_curated_training_pair).
    _write_pair(tmp_path / "training_data" / "round_00" / "accepted_pairs" / "spatial", "s1", "XY", 1)
    ctrl._touch_sample("s1_XY_z0001", status="accepted")

    pairs = ctrl._collect_prior_round_pairs()
    assert len(pairs) == 1


def test_from_scratch_training_ignores_prior(tmp_path):
    """With training_mode='from_scratch', prior pairs should not be used."""
    ctrl = _make_ctrl(tmp_path, round_num=1)
    ctrl.training_mode = "from_scratch"

    _write_pair(tmp_path / "training_data" / "round_00" / "spatial", "s1", "XY", 1)

    sl = "patch01_XY_10"
    ctrl.segmentation_manager.set(sl, np.random.rand(2, 8, 8).astype(np.float32), np.ones((8, 8), dtype=np.uint16))
    ctrl._touch_sample(sl, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([sl])

    ctrl._train_model()

    # Training should still succeed with only 1 sample.
    assert ctrl.current_round == 2
    # Phase A (2026-05-25): fallback write path now lands in `accepted/`
    # (canonical curated-pair location) instead of legacy `spatial/`
    # subdir.  The empty `spatial/`/`axial/` subdirs that used to be
    # mkdir'd unconditionally are gone.
    accepted_dir = tmp_path / "training_data" / "round_01" / "accepted"
    img_files = list(accepted_dir.glob("*_img.tif"))
    assert len(img_files) == 1


def test_cumulative_training_merges_prior_and_current(tmp_path):
    """Cumulative mode should combine prior round data with current staged data."""
    ctrl = _make_ctrl(tmp_path, round_num=1)
    ctrl.training_mode = "cumulative"

    # Prior round data (in accepted_pairs/).
    _write_pair(tmp_path / "training_data" / "round_00" / "accepted_pairs" / "spatial", "s1", "XY", 1)
    ctrl._touch_sample("s1_XY_z0001", status="accepted")

    # Current staged.
    sl = "patch01_XY_10"
    ctrl.segmentation_manager.set(sl, np.random.rand(2, 8, 8).astype(np.float32), np.ones((8, 8), dtype=np.uint16))
    ctrl._touch_sample(sl, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([sl])

    ctrl._train_model()

    assert ctrl.current_round == 2
    # Should have logged prior pairs message.
    assert any("prior pair" in line for line in ctrl.ui.training_log_output.lines)


# --- Issue 5: Session reload ---


def test_session_reload_restores_pairs(tmp_path):
    """After state reload, persisted pairs should be back in segmentation_manager."""
    ctrl = _make_ctrl(tmp_path)
    sl = "patch01_XY_5"

    # Write pair to accepted_pairs path and populate sample_records.
    pair_dir = tmp_path / "training_data" / "round_00" / "accepted_pairs" / "spatial"
    img_uri, mask_uri = _write_pair(pair_dir, "patch01", "XY", 5)
    ctrl.sample_records[sl] = {
        "sample_id": sl,
        "status": "accepted",
        "view": "XY",
        "bucket": "lateral",
        "img_uri": img_uri,
        "mask_uri": mask_uri,
    }

    # Clear in-memory data (simulating session restart).
    ctrl.segmentation_manager.data_inputs.clear()
    ctrl.segmentation_manager.data_masks.clear()

    ctrl._reload_persisted_pairs_from_disk()

    assert sl in ctrl.segmentation_manager.data_inputs
    assert sl in ctrl.segmentation_manager.data_masks


def test_session_reload_skips_missing_files(tmp_path):
    """Reload should log warning for samples with missing disk files."""
    ctrl = _make_ctrl(tmp_path)
    sl = "patch01_XY_5"
    ctrl.sample_records[sl] = {
        "sample_id": sl,
        "status": "accepted",
        "view": "XY",
        "bucket": "lateral",
        "img_uri": str(tmp_path / "nonexistent_img.tif"),
        "mask_uri": str(tmp_path / "nonexistent_mask.tif"),
    }

    ctrl._reload_persisted_pairs_from_disk()

    # Should not crash, and should not populate segmentation_manager.
    assert sl not in ctrl.segmentation_manager.data_inputs


# --- Issue 6: Review tab population ---


def test_unified_slice_list_shows_all_statuses():
    """Unified slice list should contain items for all statuses in active view."""
    ctrl = _make_ctrl(Path("/tmp/test_tabs"))
    ctrl._touch_sample("patch01_XY_1", status="predicted", view="XY")
    ctrl._touch_sample("patch01_XY_2", status="predicted", view="XY")
    ctrl._touch_sample("patch01_XY_3", status="accepted", view="XY")
    ctrl._touch_sample("patch01_XY_4", status="staged", view="XY")

    ctrl._refresh_sample_queue()

    # Unified list shows all 4 items (filtered by lateral bucket + patch01).
    assert ctrl.ui.segment_slice_data_list.count() == 4


def test_unified_list_contains_accepted_and_staged():
    """Unified slice list shows accepted and staged items with correct format."""
    ctrl = _make_ctrl(Path("/tmp/test_unified"))
    ctrl._touch_sample("patch01_XY_1", status="predicted", view="XY")
    ctrl._touch_sample("patch01_XY_2", status="accepted", view="XY")
    ctrl._touch_sample("patch01_XY_3", status="staged", view="XY")

    ctrl._refresh_sample_queue()

    # All 3 items in the unified list (filtered by lateral bucket).
    items = [ctrl.ui.segment_slice_data_list.item(i).text()
             for i in range(ctrl.ui.segment_slice_data_list.count())]
    assert len(items) == 3
    # Items sorted by status rank: staged first, then accepted, then predicted.
    assert "slice 3" in items[0] or "patch01" in items[0]  # staged
    assert "slice 2" in items[1] or "patch01" in items[1]  # accepted
    assert "slice 1" in items[2] or "patch01" in items[2]  # predicted


# --- Issue 4: Persisted pairs dedup ---


def test_prepare_dataset_reuses_persisted_pairs(tmp_path):
    """prepare_round_dataset should reuse existing disk pairs instead of re-writing."""
    tm = TrainingManager()
    seg = DummySegmentationManager()
    sl = "patch01_XY_5"

    # Write a pair to disk and provide it as persisted.
    pair_dir = tmp_path / "existing_pairs"
    img_uri, mask_uri = _write_pair(pair_dir, "patch01", "XY", 5)

    persisted = {sl: {"img_uri": img_uri, "mask_uri": mask_uri}}

    # Note: segmentation_manager has no data for this label.
    samples = tm.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=[sl],
        working_dir=str(tmp_path),
        round_id="round_00",
        persisted_pairs=persisted,
    )

    assert len(samples) == 1
    assert samples[0]["img_uri"] == img_uri
    assert samples[0]["mask_uri"] == mask_uri

    # No new files should be written to the round directory.
    round_spatial = tmp_path / "training_data" / "round_00" / "spatial"
    new_files = list(round_spatial.glob("*_img.tif"))
    assert len(new_files) == 0


def test_resolve_prior_model_checkpoint():
    """Should resolve checkpoint from previous round."""
    ctrl = _make_ctrl(Path("/tmp/test_ckpt"))
    ctrl.current_round = 2
    ctrl.model_checkpoint_paths = {
        0: {"spatial": "/models/r0_spatial.ckpt", "axial": "/models/r0_axial.ckpt"},
        1: {"spatial": "/models/r1_spatial.ckpt", "axial": "/models/r1_axial.ckpt"},
    }

    result = ctrl._resolve_prior_model_checkpoint()
    assert result["spatial"] == "/models/r1_spatial.ckpt"
    assert result["axial"] == "/models/r1_axial.ckpt"


def test_resolve_prior_model_checkpoint_round_zero():
    """Round 0 has no prior model."""
    ctrl = _make_ctrl(Path("/tmp/test_ckpt_r0"))
    ctrl.current_round = 0

    result = ctrl._resolve_prior_model_checkpoint()
    assert result == {}


# --- Duplicate acceptance handling ---


def _make_ctrl_with_viewer(tmp_path, volume_shape=(10, 16, 16)):
    """Create a controller with mock viewer for native accept tests."""
    ctrl = _make_ctrl(tmp_path)

    label_data = np.zeros(volume_shape, dtype=np.uint16)
    img_data = np.random.rand(*volume_shape).astype(np.float32)

    class MockLayer:
        def __init__(self, name, data, cls_name="Image"):
            self.name = name
            self.data = data
            self._cls_name = cls_name

        @property
        def __class__(self):
            class _C:
                pass
            c = _C()
            c.__name__ = self._cls_name
            return c

    class MockLayers:
        def __init__(self, layers):
            self._layers = layers

        def __iter__(self):
            return iter(self._layers)

        def __contains__(self, name):
            return any(l.name == name for l in self._layers)

    class MockViewer:
        pass

    img_layer = MockLayer("patch01_ch0_XY_view", img_data, "Image")
    labels_layer = MockLayer("patch01_work_labels_lateral_3d", label_data, "Labels")
    viewer = MockViewer()
    viewer.layers = MockLayers([img_layer, labels_layer])
    ctrl.viewer = viewer

    return ctrl, labels_layer, img_layer


def test_accept_skips_unchanged_mask(tmp_path):
    """If mask hasn't changed and is already accepted, accept should skip."""
    ctrl, labels_layer, _ = _make_ctrl_with_viewer(tmp_path)

    # Put some labels at slice 3.
    labels_layer.data[3, 5:10, 5:10] = 1

    # First accept — should succeed normally.
    result1 = ctrl._accept_native_annotation("XY", slice_index=3)
    assert result1["ok"] is True
    assert result1.get("skipped") is not True
    assert ctrl.sample_records["patch01_XY_3"]["status"] == "accepted"

    # Second accept with same mask — should skip.
    result2 = ctrl._accept_native_annotation("XY", slice_index=3)
    assert result2["ok"] is True
    assert result2.get("skipped") is True
    assert "unchanged" in result2["message"].lower()
    # Status should still be accepted.
    assert ctrl.sample_records["patch01_XY_3"]["status"] == "accepted"


def test_accept_skips_unchanged_staged_mask(tmp_path):
    """If mask hasn't changed and is already staged, accept should skip."""
    ctrl, labels_layer, _ = _make_ctrl_with_viewer(tmp_path)

    labels_layer.data[3, 5:10, 5:10] = 1

    # Accept then stage.
    ctrl._accept_native_annotation("XY", slice_index=3)
    ctrl._touch_sample("patch01_XY_3", status="staged", view="XY")

    # Re-accept with same mask — should skip, preserving staged.
    result = ctrl._accept_native_annotation("XY", slice_index=3)
    assert result["ok"] is True
    assert result.get("skipped") is True
    assert ctrl.sample_records["patch01_XY_3"]["status"] == "staged"


def test_accept_preserves_staged_status_on_mask_change(tmp_path):
    """If mask changed but sample is staged, update data but keep staged status."""
    ctrl, labels_layer, _ = _make_ctrl_with_viewer(tmp_path)

    # Initial accept.
    labels_layer.data[3, 5:10, 5:10] = 1
    ctrl._accept_native_annotation("XY", slice_index=3)

    # Stage the sample.
    ctrl._touch_sample("patch01_XY_3", status="staged", view="XY")
    assert ctrl.sample_records["patch01_XY_3"]["status"] == "staged"

    # Modify the mask and re-accept.
    labels_layer.data[3, 5:10, 5:10] = 2  # Different label values.
    result = ctrl._accept_native_annotation("XY", slice_index=3)

    assert result["ok"] is True
    assert result.get("skipped") is not True
    # Status should still be staged (not demoted to accepted).
    assert ctrl.sample_records["patch01_XY_3"]["status"] == "staged"
    # Data should be updated in segmentation manager.
    updated_mask = ctrl.segmentation_manager.get_masks("patch01_XY_3")
    assert np.max(updated_mask) == 2


def test_accept_updates_accepted_on_mask_change(tmp_path):
    """If mask changed and sample is accepted, re-accept with new data."""
    ctrl, labels_layer, _ = _make_ctrl_with_viewer(tmp_path)

    # Initial accept.
    labels_layer.data[3, 5:10, 5:10] = 1
    ctrl._accept_native_annotation("XY", slice_index=3)
    assert ctrl.sample_records["patch01_XY_3"]["status"] == "accepted"

    # Modify the mask and re-accept.
    labels_layer.data[3, 5:10, 5:10] = 3
    result = ctrl._accept_native_annotation("XY", slice_index=3)

    assert result["ok"] is True
    assert result.get("skipped") is not True
    assert ctrl.sample_records["patch01_XY_3"]["status"] == "accepted"
    updated_mask = ctrl.segmentation_manager.get_masks("patch01_XY_3")
    assert np.max(updated_mask) == 3


# --- Color-coded review queue items ---


def test_unified_list_items_get_color_coded():
    """Items in the unified slice list should have foreground color based on status."""
    ctrl = _make_ctrl(Path("/tmp/test_colors"))

    ctrl._touch_sample("patch01_XY_1", status="predicted", view="XY")
    ctrl._touch_sample("patch01_XY_2", status="accepted", view="XY")
    ctrl._touch_sample("patch01_XY_3", status="staged", view="XY")

    ctrl._refresh_sample_queue()

    # All 3 items in unified list, each with foreground color set.
    count = ctrl.ui.segment_slice_data_list.count()
    assert count == 3
    for i in range(count):
        item = ctrl.ui.segment_slice_data_list.item(i)
        assert item._foreground is not None


# --- View Results navigation ---


def test_parse_queue_selected_sample_id_formats():
    """_parse_queue_selected_sample_id handles both raw and monospace formats."""
    ctrl = _make_ctrl(Path("/tmp/test_parse"))
    sl = "patch01_XY_5"
    ctrl._touch_sample(sl, status="predicted", view="XY")

    # Raw sample_id format (direct match).
    ctrl.ui.segment_slice_data_list.addItem(sl)
    ctrl.ui.segment_slice_data_list._selected = [_Item(sl)]
    assert ctrl._parse_queue_selected_sample_id() == sl

    # New monospace format.
    formatted = "XY  slice 5      patch01          "
    ctrl.ui.segment_slice_data_list._selected = [_Item(formatted)]
    assert ctrl._parse_queue_selected_sample_id() == sl


# --- Predicted-state persistence (regression for "predicted slices lost on restart") ---


def test_run_segmentation_calls_schedule_state_save():
    """_run_segmentation must call _schedule_state_save() so predicted records persist.

    Regression: previously only Accept/Stage/Train triggered state save, so predicted
    sample_records existed only in memory until the user accepted at least one slice.
    Closing napari before any accept dropped all predicted records (user-visible:
    box list lost predicted slices on restart).
    """
    import inspect
    src = inspect.getsource(TrainingAnnotationController._run_segmentation)
    assert "_schedule_state_save" in src, (
        "_run_segmentation must call _schedule_state_save() so predicted slices "
        "persist to hitl_state.json across napari restarts."
    )


def test_create_empty_mask_seed_calls_schedule_state_save():
    """_create_empty_mask_seed must persist its 'predicted' record (same regression class)."""
    import inspect
    src = inspect.getsource(TrainingAnnotationController._create_empty_mask_seed)
    assert "_schedule_state_save" in src, (
        "_create_empty_mask_seed must call _schedule_state_save() so manual seeds "
        "persist to hitl_state.json across napari restarts."
    )


def test_predicted_record_round_trips_through_save_load(tmp_path):
    """Predicted sample_records must survive _save_internal_state / load cycle."""
    ctrl = _make_ctrl(tmp_path)

    slice_label = "patch01_XY_42"
    ctrl._touch_sample(slice_label, status="predicted", view="XY")
    assert ctrl.sample_records[slice_label]["status"] == "predicted"

    # Persist to disk (this is what _schedule_state_save eventually triggers).
    ctrl._save_internal_state()

    state_file = tmp_path / "hitl_state.json"
    assert state_file.exists(), "hitl_state.json should be written"

    persisted = json.loads(state_file.read_text())
    assert "sample_records" in persisted
    assert slice_label in persisted["sample_records"]
    assert persisted["sample_records"][slice_label]["status"] == "predicted"
    assert persisted["sample_records"][slice_label]["bucket"] == "lateral"

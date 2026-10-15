import json
import numpy as np

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

    def count(self):
        return len(self._items)

    def item(self, i):
        return self._items[i]

    def addItem(self, text):
        self._items.append(_Item(text))

    def clear(self):
        self._items = []


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


class DummySegmentationManager:
    def __init__(self):
        self._inputs = {}
        self._masks = {}

    def set(self, slice_label, inp, mask):
        self._inputs[slice_label] = inp
        self._masks[slice_label] = mask

    def get_input_stack(self, slice_label):
        return self._inputs.get(slice_label)

    def get_masks(self, slice_label):
        return self._masks.get(slice_label)


def _make_train_ctrl(tmp_path, *, config=None, seg_manager=None, training_manager=None):
    """Shared helper for training round flow tests."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.training_manager = training_manager or TrainingManager()
    ctrl.segmentation_manager = seg_manager or DummySegmentationManager()
    ctrl.working_dir = str(tmp_path)
    ctrl.current_round = 0
    ctrl.current_config = config or {"model_name": "epi", "model_type": "cyto3", "epochs": 120}
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.round_backend_info = {}
    ctrl.loaded_patch_name = "stack01"
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl.axial_plane = "XZ"
    ctrl.sample_records = {}
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
    ui.accepted_queue_list = _ListWidget()
    ui.staged_queue_list = _ListWidget()
    ui.sample_queue_list = ui.accepted_queue_list
    ctrl.ui = ui

    # Fake viewer with empty layers list for _iter_work_label_layers.
    class _FakeViewer:
        class _Layers(list):
            pass
        layers = _Layers()
    ctrl.viewer = _FakeViewer()

    return ctrl


def test_train_model_round_updates_state_and_persists_artifacts(tmp_path):
    ctrl = _make_train_ctrl(tmp_path)

    slice_label = "stack01_XY_12"
    ctrl.segmentation_manager.set(
        slice_label,
        np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16),
    )
    ctrl._touch_sample(slice_label, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([slice_label])

    ctrl._train_model()

    assert ctrl.current_round == 1
    assert "lateral_0" in ctrl.training_data_by_round
    assert len(ctrl.training_data_by_round["lateral_0"]) == 1
    assert "lateral_0" in ctrl.model_checkpoint_paths
    assert ctrl.ui.training_progress_bar.value == 100
    assert len(ctrl.ui.training_log_output.lines) >= 1
    assert any("Trained round_00" in line for line in ctrl.ui.training_log_output.lines)
    inference_artifacts_dir = tmp_path / "artifacts" / "inference"
    assert (inference_artifacts_dir / "epi_spatial_round_00_spatial.json").exists()


def test_train_model_lateral_only_trains_lateral_slices(tmp_path):
    """Training in lateral mode should only use lateral (XY) staged slices."""
    ctrl = _make_train_ctrl(tmp_path, config={"model_name": "epi", "model_type": "cyto3", "epochs": 80})
    ctrl.loaded_patch_name = "stackmix"
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"

    xy = "stackmix_XY_11"
    xz = "stackmix_XZ_4"
    ctrl.segmentation_manager.set(xy, np.random.rand(2, 16, 16).astype(np.float32), np.ones((16, 16), dtype=np.uint16))
    ctrl.segmentation_manager.set(xz, np.random.rand(2, 16, 16).astype(np.float32), np.ones((16, 16), dtype=np.uint16))
    ctrl._touch_sample(xy, status="staged", view="XY")
    ctrl._touch_sample(xz, status="staged", view="XZ")

    ctrl._train_model()

    # Should only have spatial checkpoint (lateral mode filters out axial).
    checkpoints = ctrl.model_checkpoint_paths.get("lateral_0")
    assert checkpoints is not None
    # Single-view training returns a path string, not a dict.
    if isinstance(checkpoints, dict):
        assert "spatial" in checkpoints
        assert "axial" not in checkpoints
    else:
        assert isinstance(checkpoints, str)


def test_run_benchmark_writes_summary(tmp_path):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.working_dir = str(tmp_path)
    ctrl.losses_by_round = {
        0: {"spatial": [0.7, 0.5], "axial": [0.8, 0.6]},
        1: {"spatial": [0.6, 0.4], "axial": [0.7, 0.5]},
    }
    ctrl.training_data_by_round = {
        0: [{"slice_index": 1}, {"slice_index": 2}],
        1: [{"slice_index": 3}],
    }
    ctrl.model_checkpoint_paths = {0: {"spatial": "a", "axial": "b"}, 1: {"spatial": "c", "axial": "d"}}
    ctrl.round_backend_info = {
        0: {"training": {"spatial": {"mode": "placeholder"}}, "inference": {}, "failed_views": []},
        1: {"training": {"spatial": {"mode": "placeholder"}}, "inference": {}, "failed_views": []},
    }

    class UI:
        pass

    ui = UI()
    ui.training_log_output = _LogOutput()
    ctrl.ui = ui

    ctrl._run_benchmark()

    summary_path = tmp_path / "artifacts" / "benchmarks" / "round_benchmark_summary.json"
    assert summary_path.exists()
    assert any("Best round" in line for line in ui.training_log_output.lines)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "per_view_final_loss" in payload["rows"][0]
    assert "training_backend" in payload["rows"][0]


def test_train_model_single_view_completes_successfully(tmp_path):
    """Training in lateral mode with only XY slices should complete without axial errors."""
    ctrl = _make_train_ctrl(
        tmp_path,
        config={"model_name": "epi", "model_type": "cyto3", "epochs": 80},
    )
    ctrl.loaded_patch_name = "stackmix"
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"

    xy1 = "stackmix_XY_11"
    xy2 = "stackmix_XY_15"
    ctrl.segmentation_manager.set(xy1, np.random.rand(2, 16, 16).astype(np.float32), np.ones((16, 16), dtype=np.uint16))
    ctrl.segmentation_manager.set(xy2, np.random.rand(2, 16, 16).astype(np.float32), np.ones((16, 16), dtype=np.uint16))
    ctrl._touch_sample(xy1, status="staged", view="XY")
    ctrl._touch_sample(xy2, status="staged", view="XY")

    ctrl._train_model()

    assert ctrl.current_round == 1
    assert "lateral_0" in ctrl.round_backend_info
    # No axial failures since only lateral slices were trained.
    failed = ctrl.round_backend_info["lateral_0"].get("failed_views", [])
    assert not any(entry.get("view_type") == "axial" for entry in failed)


def test_train_model_promotes_latest_checkpoint_for_curation(tmp_path):
    checkpoint_path = tmp_path / "models" / "round_00" / "epi_spatial_round_00"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    def trainer_backend(**kwargs):
        return {
            "checkpoint_path": str(checkpoint_path),
            "losses": [0.8, 0.4],
            "mode": "custom_backend",
            "view_type": kwargs["view_type"],
        }

    ctrl = _make_train_ctrl(
        tmp_path,
        config={"model_name": "epi", "model_type": "cyto3", "epochs": 40},
        training_manager=TrainingManager(backend_trainer=trainer_backend),
    )
    ctrl.loaded_patch_name = "stackxy"

    xy = "stackxy_XY_7"
    ctrl.segmentation_manager.set(xy, np.random.rand(2, 16, 16).astype(np.float32), np.ones((16, 16), dtype=np.uint16))
    ctrl._touch_sample(xy, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([xy])

    ctrl._train_model()

    assert ctrl.active_curation_checkpoints["lateral"] == str(checkpoint_path)
    assert ctrl.current_config.get("pretrained_model_path") == str(checkpoint_path)
    assert any("Active lateral model set" in line for line in ctrl.ui.training_log_output.lines)


def test_persist_curated_training_pair_writes_image_mask_files(tmp_path):
    """Fresh workspace + persist: auto-bootstraps default run + writes
    to NEW layout (artifacts/step2/runs/default/axial/rounds/round_03/accepted/).

    Updated 2026-05-24: user reported the legacy fallback was firing on
    fresh workspaces, creating a stray training_data/ folder.  The fix
    in _persist_curated_training_pair auto-bootstraps a default run on
    UNKNOWN layout instead of falling back to legacy.  This test now
    verifies the CORRECT behavior."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.segmentation_manager = DummySegmentationManager()
    ctrl.working_dir = str(tmp_path)
    ctrl.current_round = 3
    ctrl.sample_records = {}

    slice_label = "stack_patch_XZ_9"
    ctrl.segmentation_manager.set(
        slice_label,
        np.random.rand(2, 12, 14).astype(np.float32),
        np.ones((12, 14), dtype=np.uint16),
    )

    out = ctrl._persist_curated_training_pair(slice_label)

    assert out["img_uri"]
    assert out["mask_uri"]
    # Phase B.2 (2026-05-25) flat layout:
    # artifacts/step2/runs/default/axial/round_03/accepted/
    # Pre-B.2 was `axial/rounds/round_03/accepted/` (split).
    new_layout_dir = (
        tmp_path / "artifacts" / "step2" / "runs" / "default"
        / "axial" / "round_03" / "accepted"
    )
    assert new_layout_dir.exists(), (
        f"expected new-layout dir {new_layout_dir} to exist after persist; "
        f"saw out_dir={out}"
    )
    assert (new_layout_dir / "stack_patch_XZ_z0009_img.tif").exists()
    assert (new_layout_dir / "stack_patch_XZ_z0009_mask.tif").exists()
    assert ctrl.sample_records[slice_label]["img_uri"].endswith("_img.tif")
    assert ctrl.sample_records[slice_label]["mask_uri"].endswith("_mask.tif")
    # NEGATIVE assertion: legacy fallback should NOT have fired
    assert not (tmp_path / "training_data").exists(), (
        "legacy training_data/ folder should NOT be created on fresh workspace "
        "with the new auto-bootstrap fix in place"
    )
    # Verify the bootstrap also created the run manifest + state JSON.
    # State JSON lives at workspace root (per _paths.state_json_path),
    # NOT under artifacts/step2/.
    assert (
        tmp_path / "artifacts" / "step2" / "runs" / "default" / "run_manifest.json"
    ).exists()
    assert (tmp_path / "step2_state.json").exists()


def test_insert_slice_mask_with_unique_labels_preserves_existing_predictions():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    class _FV:
        class _L(list):
            pass
        layers = _L()
    ctrl.viewer = _FV()

    volume = np.zeros((4, 8, 8), dtype=np.uint16)
    # Existing predicted slice with foreground IDs 1..2
    volume[1, 1:4, 1:4] = 1
    volume[1, 4:7, 4:7] = 2
    old_slice = volume[1].copy()

    new_mask = np.zeros((8, 8), dtype=np.uint16)
    new_mask[2:6, 2:6] = 1  # cellpose-style local IDs restarting at 1
    next_label = ctrl._insert_slice_mask_with_unique_labels(volume, new_mask, "XY", 3)

    # Existing segmented slice must be preserved.
    assert np.array_equal(volume[1], old_slice)
    # New slice must be written and relabeled above prior max (2).
    assert int(volume[3].max()) >= 3
    assert int(next_label) == int(volume.max()) + 1


def test_sync_selected_label_bumps_to_next_unique():
    """After inserting masks, selected_label should be max(volume)+1."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    class _FV:
        class _L(list):
            pass
        layers = _L()
    ctrl.viewer = _FV()

    class _FakeLabelsLayer:
        def __init__(self, data):
            self.data = data
            self.selected_label = 1

    volume = np.zeros((4, 8, 8), dtype=np.uint16)
    volume[0, 2:5, 2:5] = 1
    volume[0, 5:7, 5:7] = 2
    volume[1, 1:3, 1:3] = 3

    layer = _FakeLabelsLayer(volume)
    ctrl._sync_selected_label_to_volume(layer)

    # selected_label should be 4 (max=3, +1)
    assert layer.selected_label == 4

    # After erasing slice 1 and re-inserting, label should still be safe.
    new_mask = np.zeros((8, 8), dtype=np.uint16)
    new_mask[3:6, 3:6] = 1
    ctrl._insert_slice_mask_with_unique_labels(volume, new_mask, "XY", 2)
    layer.data = volume
    ctrl._sync_selected_label_to_volume(layer)

    # The new mask should have been relabeled to 4 (max was 3 before clear),
    # so selected_label should now be max(volume)+1 = 5.
    assert layer.selected_label == int(volume.max()) + 1
    assert layer.selected_label >= 5


def test_apply_active_view_to_viewer_preserves_ndisplay_by_default():
    class _Dims:
        def __init__(self):
            self.ndim = 3
            self.ndisplay = 3
            self.order = (0, 1, 2)
            self.nsteps = (10, 20, 30)
            self.current_step = [0, 0, 0]

        def set_current_step(self, axis, value):
            self.current_step[int(axis)] = int(value)

    class _Viewer:
        def __init__(self):
            self.dims = _Dims()

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.viewer = _Viewer()
    ctrl.selected_view = "XY"

    ctrl.apply_active_view_to_viewer(slice_index=4)

    assert ctrl.viewer.dims.ndisplay == 3
    assert tuple(ctrl.viewer.dims.order) == (0, 1, 2)
    assert int(ctrl.viewer.dims.current_step[0]) == 4


# ============================================================================
# Phase B.1 / B.1.1 — Mode=Both DRIFT regression tests (2026-05-25)
# Per audit recommendations B.1-T1..T10.  Verifies the 4 silent-data-
# overwrite drift sites are FIXED, AND the audit-found loss-plot label
# leak + persistence fallback bugs are CLOSED.
# ============================================================================

def _stage_mixed_view(ctrl, monkeypatch, xy_label="stk_XY_5", xz_label="stk_XZ_3"):
    """Helper: stage one XY + one XZ slice for a Mode=Both training run.

    Production's `_sync_data_to_add_list` filters by `active_bucket`
    (matching `_bucket_for_view(self.selected_view)`).  Tests need
    BOTH views fed to `_run_training_round` so the per-view loop fires
    twice — monkeypatch sync to be a no-op so the list we manually
    populated survives the call from `_train_model`.
    """
    import numpy as np
    ctrl.segmentation_manager.set(
        xy_label, np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16))
    ctrl.segmentation_manager.set(
        xz_label, np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16))
    ctrl._touch_sample(xy_label, status="staged", view="XY")
    ctrl._touch_sample(xz_label, status="staged", view="XZ")
    ctrl.ui.data_to_add_list = _ListWidget([xy_label, xz_label])
    # Bypass production bucket-filter so the test reaches Mode=Both
    monkeypatch.setattr(ctrl, "_sync_data_to_add_list", lambda force=False: None)
    return xy_label, xz_label


def test_b1_mode_both_per_bucket_state_keys(tmp_path, monkeypatch):
    """B.1-T1: Mode=Both must write per-bucket composite keys to
    training_data_by_round (not lump under ONE key)."""
    ctrl = _make_train_ctrl(tmp_path)
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    _stage_mixed_view(ctrl, monkeypatch)

    ctrl._train_model()

    # Both buckets present as separate keys
    assert "lateral_0" in ctrl.training_data_by_round
    assert "axial_0" in ctrl.training_data_by_round
    # Each bucket's samples are view-filtered (no lateral pollution in axial entry)
    lateral_views = {s.get("view") for s in ctrl.training_data_by_round["lateral_0"]}
    axial_views = {s.get("view") for s in ctrl.training_data_by_round["axial_0"]}
    assert lateral_views == {"XY"}, f"lateral_0 should be XY-only, got {lateral_views}"
    assert axial_views.issubset({"XZ", "YZ"}), \
        f"axial_0 should be XZ/YZ-only, got {axial_views}"


def test_b1_mode_both_increments_both_counters(tmp_path, monkeypatch):
    """B.1-T2: Mode=Both must increment BOTH per-bucket counters."""
    ctrl = _make_train_ctrl(tmp_path)
    ctrl.current_round_by_view = {"lateral": 2, "axial": 0}
    _stage_mixed_view(ctrl, monkeypatch)

    ctrl._train_model()

    assert ctrl.current_round_by_view["lateral"] == 3, \
        f"lateral should advance 2→3, got {ctrl.current_round_by_view['lateral']}"
    assert ctrl.current_round_by_view["axial"] == 1, \
        f"axial should advance 0→1, got {ctrl.current_round_by_view['axial']}"


def test_b1_lateral_only_does_not_advance_axial(tmp_path):
    """B.1-T5: lateral-only training must NOT advance axial counter."""
    import numpy as np
    ctrl = _make_train_ctrl(tmp_path)
    ctrl.current_round_by_view = {"lateral": 0, "axial": 5}

    xy = "stk_XY_5"
    ctrl.segmentation_manager.set(
        xy, np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16))
    ctrl._touch_sample(xy, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([xy])

    ctrl._train_model()

    assert ctrl.current_round_by_view["lateral"] == 1
    assert ctrl.current_round_by_view["axial"] == 5, \
        "axial counter must not move when only lateral trained"


def test_b1_mode_both_per_bucket_losses_and_checkpoints(tmp_path, monkeypatch):
    """B.1-T7: Mode=Both populates per-bucket entries in losses_by_round
    + model_checkpoint_paths.  The H-1 bug lumped them into ONE key.
    """
    ctrl = _make_train_ctrl(tmp_path)
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    _stage_mixed_view(ctrl, monkeypatch)

    ctrl._train_model()

    assert "lateral_0" in ctrl.losses_by_round
    assert "axial_0" in ctrl.losses_by_round
    assert "lateral_0" in ctrl.model_checkpoint_paths
    assert "axial_0" in ctrl.model_checkpoint_paths


def test_b1_run_training_work_snapshots_threaded(tmp_path, monkeypatch):
    """B.1-T6: launcher threads selected_view + current_round_by_view
    snapshots into _run_training_work kwargs.  Without these snapshots,
    DRIFT-1 race surface persists.
    """
    ctrl = _make_train_ctrl(tmp_path)
    ctrl.current_round_by_view = {"lateral": 7, "axial": 3}
    ctrl.selected_view = "XY"

    import numpy as np
    xy = "stk_XY_5"
    ctrl.segmentation_manager.set(
        xy, np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16))
    ctrl._touch_sample(xy, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([xy])

    captured = {}
    original = ctrl._run_training_work
    def _capturing(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(ctrl, "_run_training_work", _capturing)

    ctrl._train_model()

    assert "selected_view_snapshot" in captured, \
        "DRIFT-1 fix incomplete: selected_view_snapshot not in work_kwargs"
    assert "current_round_by_view_snapshot" in captured, \
        "DRIFT-1 fix incomplete: current_round_by_view_snapshot not in work_kwargs"
    assert captured["selected_view_snapshot"] == "XY"
    assert captured["current_round_by_view_snapshot"] == {"lateral": 7, "axial": 3}


def test_b1_mode_both_loss_plot_labels_per_bucket(tmp_path, monkeypatch):
    """B.1-T9 (audit 1+2 HIGH-1): loss-plot label leak.  Pre-fix,
    Mode=Both gave both curves the SAME bucket key (leaked loop var).
    Post-fix, each curve's label reflects ITS OWN bucket.
    """
    ctrl = _make_train_ctrl(tmp_path)
    ctrl.current_round_by_view = {"lateral": 2, "axial": 4}
    _stage_mixed_view(ctrl, monkeypatch)

    class _CapturingLossPlot:
        def __init__(self):
            self.curves = []
        def begin_live_round(self, round_id):
            pass
        def append_live_point(self, epoch, loss):
            pass
        def cancel_live_round(self):
            pass
        def finish_live_round(self):
            pass
        def add_round_curve(self, label, losses):
            self.curves.append((label, list(losses)))
    ctrl.ui.loss_plot = _CapturingLossPlot()

    ctrl._train_model()

    labels = [c[0] for c in ctrl.ui.loss_plot.curves]
    # Each label should embed its OWN bucket key.  Pre-fix bug:
    # both labels start with the same bucket prefix (whichever loop
    # last set round_key — usually axial).
    assert any(l.startswith("lateral_2") for l in labels), \
        f"lateral curve label missing 'lateral_2' prefix; got {labels}"
    assert any(l.startswith("axial_4") for l in labels), \
        f"axial curve label missing 'axial_4' prefix; got {labels}"


def test_b1_per_bucket_metadata_uses_per_bucket_round_idx(tmp_path):
    """B.1-T3 (DRIFT-4): `_write_per_bucket_metadata_for_active_run`
    must use per-bucket round_idx, not selected_view's round.  Setup:
    counters lateral=3, axial=1 (post-increment, just-trained rounds
    were lateral=2, axial=0).  Each bucket's metadata.json must land
    at its own round_NN/, not at one shared dir.
    """
    import json as _json
    ctrl = _make_train_ctrl(tmp_path)
    # Mark layout as new (controller uses _step2_paths.is_new_layout_state)
    from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
    state_path = _step2_paths.state_json_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(_json.dumps({
        "schema_version": 3,
        "active_run_id": "default",
        "runs": {"default": {"training_data_by_round": {}}},
    }), encoding="utf-8")
    ctrl.active_run_id = "default"
    # Simulate post-train state (counters incremented to 3 and 1)
    ctrl.current_round_by_view = {"lateral": 3, "axial": 1}
    ctrl.training_data_by_round = {
        "lateral_2": [{
            "stack_id": "stk", "view": "XY", "slice_index": 5,
            "img_uri": "x", "mask_uri": "y",
        }],
        "axial_0": [{
            "stack_id": "stk", "view": "XZ", "slice_index": 3,
            "img_uri": "x", "mask_uri": "y",
        }],
    }
    ctrl.sample_records = {
        "stk_XY_5": {"stack_id": "stk", "view": "XY", "slice_index": 5,
                     "round_idx": 2},
        "stk_XZ_3": {"stack_id": "stk", "view": "XZ", "slice_index": 3,
                     "round_idx": 0},
    }

    ctrl._write_per_bucket_metadata_for_active_run("default")

    lateral_meta = _step2_paths.round_metadata_path(tmp_path, "default", "lateral", 2)
    axial_meta = _step2_paths.round_metadata_path(tmp_path, "default", "axial", 0)
    assert lateral_meta.is_file(), \
        f"Expected lateral metadata at round_02, missing: {lateral_meta}"
    assert axial_meta.is_file(), \
        f"Expected axial metadata at round_00, missing: {axial_meta}"
    # Selected-indices files at per-bucket round_idx
    lateral_sel = _step2_paths.round_selected_indices_path(tmp_path, "default", "lateral", 2)
    axial_sel = _step2_paths.round_selected_indices_path(tmp_path, "default", "axial", 0)
    assert lateral_sel.is_file()
    assert axial_sel.is_file()


def test_b1_hitl_persistence_composite_key_fallback(tmp_path):
    """B.1-T10 (audit 2 HIGH-2): legacy persistence path must read
    composite keys (post-B.1 controller writes only those).  Pre-fix,
    bare-int / str-int only → empty metadata + selected_indices files.
    """
    from napari_epicycle.io_registry import HITLPersistenceService
    svc = HITLPersistenceService()

    training_data_by_round = {
        "lateral_0": [{
            "stack_id": "stk", "view": "XY", "slice_index": 5,
            "img_uri": "/x", "mask_uri": "/y",
            "dataset_id": "ds1",
        }],
        "axial_0": [{
            "stack_id": "stk", "view": "XZ", "slice_index": 3,
            "img_uri": "/x", "mask_uri": "/y",
            "dataset_id": "ds1",
        }],
    }

    metadata = svc._build_round_metadata(
        current_round=0,
        current_config={"model_name": "epi"},
        training_data_by_round=training_data_by_round,
    )
    assert len(metadata.samples) == 2, \
        f"Composite-key fallback failed: expected 2 samples, got {len(metadata.samples)}"

    sel_record = svc._build_selected_indices_record(
        round_id="round_00",
        training_data_by_round=training_data_by_round,
        current_round=0,
    )
    # Per existing schema: XY → "XY" bucket, XZ → "XZ"
    xy_sel = sel_record.selections.get("XY", {})
    xz_sel = sel_record.selections.get("XZ", {})
    assert "stk" in xy_sel and 5 in xy_sel["stk"], \
        f"XY selection missing post-fallback: {xy_sel}"
    assert "stk" in xz_sel and 3 in xz_sel["stk"], \
        f"XZ selection missing post-fallback: {xz_sel}"


# ============================================================================
# Phase C.5 (2026-05-26) — Mode=Both READ-side DRIFT fixes
# Audit 1 HIGH-1: `_persist_curated_training_pair` round_idx must
#   come from slice's bucket counter, NOT `_current_round()`
#   (selected_view's counter).
# Audit 3 HIGH-2: `_collect_prior_round_pairs(bucket=...)` +
#   `_resolve_prior_model_checkpoint(bucket=...)` must take explicit
#   bucket arg + use that bucket's own counter.
# ============================================================================

class TestC5PersistCuratedPairBucketCounter:
    def test_persist_xz_uses_axial_counter(self, tmp_path):
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        state_path = _step2_paths.state_json_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "schema_version": 3, "active_run_id": "default",
            "runs": {"default": {}}
        }), encoding="utf-8")

        ctrl = _make_train_ctrl(tmp_path)
        ctrl.active_run_id = "default"
        ctrl.current_round_by_view = {"lateral": 5, "axial": 2}
        ctrl.selected_view = "XY"  # selected = lateral (wrong counter pre-fix)

        slice_label = "stk_XZ_42"
        ctrl.segmentation_manager.set(
            slice_label,
            np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16),
        )

        ctrl._persist_curated_training_pair(slice_label)
        # Post-fix uses axial's counter (2), not lateral's (5)
        expected = (
            tmp_path / "artifacts" / "step2" / "runs" / "default"
            / "axial" / "round_02" / "accepted"
        )
        wrong = (
            tmp_path / "artifacts" / "step2" / "runs" / "default"
            / "axial" / "round_05" / "accepted"
        )
        assert expected.exists()
        assert not wrong.exists(), \
            f"Pre-fix bug surfaced: wrote to {wrong}"

    def test_persist_xy_uses_lateral_counter_when_selected_view_xz(self, tmp_path):
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        state_path = _step2_paths.state_json_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "schema_version": 3, "active_run_id": "default",
            "runs": {"default": {}}
        }), encoding="utf-8")

        ctrl = _make_train_ctrl(tmp_path)
        ctrl.active_run_id = "default"
        ctrl.current_round_by_view = {"lateral": 5, "axial": 2}
        ctrl.selected_view = "XZ"  # selected = axial, but persisting XY → lateral

        slice_label = "stk_XY_42"
        ctrl.segmentation_manager.set(
            slice_label,
            np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16),
        )

        ctrl._persist_curated_training_pair(slice_label)
        expected = (
            tmp_path / "artifacts" / "step2" / "runs" / "default"
            / "lateral" / "round_05" / "accepted"
        )
        assert expected.exists()


class TestC5CollectPriorPairsPerBucket:
    def test_collect_per_bucket_uses_each_buckets_counter(self, tmp_path):
        import tifffile as _tf
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        # Plant lateral 0..2 + axial 0 (each with 1 pair)
        for r in range(3):
            d = _step2_paths.round_accepted_dir(tmp_path, "default", "lateral", r)
            d.mkdir(parents=True)
            _tf.imwrite(
                str(d / f"stk_XY_z{r:04d}_img.tif"),
                np.random.rand(2, 16, 16).astype(np.float32))
            _tf.imwrite(
                str(d / f"stk_XY_z{r:04d}_mask.tif"),
                np.ones((16, 16), dtype=np.uint16))
        d_ax = _step2_paths.round_accepted_dir(tmp_path, "default", "axial", 0)
        d_ax.mkdir(parents=True)
        _tf.imwrite(
            str(d_ax / "stk_XZ_z0000_img.tif"),
            np.random.rand(2, 16, 16).astype(np.float32))
        _tf.imwrite(
            str(d_ax / "stk_XZ_z0000_mask.tif"),
            np.ones((16, 16), dtype=np.uint16))

        state_path = _step2_paths.state_json_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "schema_version": 3, "active_run_id": "default",
            "runs": {"default": {}}
        }), encoding="utf-8")

        ctrl = _make_train_ctrl(tmp_path)
        ctrl.active_run_id = "default"
        ctrl.current_round_by_view = {"lateral": 3, "axial": 1}
        ctrl.selected_view = "XY"  # doesn't matter for explicit-arg
        # B-lite (2026-09-07): only accepted/staged/committed slices are
        # inherited by cumulative training.
        for r in range(3):
            ctrl._touch_sample(f"stk_XY_z{r:04d}", status="accepted")
        ctrl._touch_sample("stk_XZ_z0000", status="accepted")

        lat = ctrl._collect_prior_round_pairs(bucket="lateral")
        ax = ctrl._collect_prior_round_pairs(bucket="axial")
        assert len(lat) == 3
        assert len(ax) == 1
        assert all(p["view_group"] == "spatial" for p in lat)
        assert all(p["view_group"] == "axial" for p in ax)


class TestC5ResolvePriorCheckpointPerBucket:
    def test_resolve_lateral_returns_only_spatial_key(self, tmp_path):
        ctrl = _make_train_ctrl(tmp_path)
        ctrl.current_round_by_view = {"lateral": 3, "axial": 1}
        ctrl.model_checkpoint_paths = {
            "lateral_2": "/path/lat",
            "axial_0": "/path/ax",
        }
        result = ctrl._resolve_prior_model_checkpoint(bucket="lateral")
        assert result == {"spatial": "/path/lat"}, \
            f"Expected only spatial key; got {result}"

    def test_resolve_axial_returns_only_axial_key(self, tmp_path):
        ctrl = _make_train_ctrl(tmp_path)
        ctrl.current_round_by_view = {"lateral": 3, "axial": 1}
        ctrl.model_checkpoint_paths = {
            "lateral_2": "/path/lat",
            "axial_0": "/path/ax",
        }
        result = ctrl._resolve_prior_model_checkpoint(bucket="axial")
        assert result == {"axial": "/path/ax"}, \
            f"Expected only axial key; got {result}"

    def test_resolve_round_0_returns_empty(self, tmp_path):
        ctrl = _make_train_ctrl(tmp_path)
        ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
        assert ctrl._resolve_prior_model_checkpoint(bucket="lateral") == {}
        assert ctrl._resolve_prior_model_checkpoint(bucket="axial") == {}

    def test_resolve_no_bucket_arg_preserves_legacy_behavior(self, tmp_path):
        ctrl = _make_train_ctrl(tmp_path)
        ctrl.current_round_by_view = {"lateral": 3, "axial": 1}
        ctrl.selected_view = "XY"
        ctrl.model_checkpoint_paths = {"lateral_2": "/path/lat"}
        result = ctrl._resolve_prior_model_checkpoint()
        # Legacy no-arg call: wraps as both spatial + axial
        assert result == {"spatial": "/path/lat", "axial": "/path/lat"}

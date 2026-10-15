"""Release-hardening #5 + #6.

#5 — `prepare_round_dataset` dropped prior-round pairs whose files are
     missing on a module logger the UI never shows: the log says
     "prior=12" while training runs on 8.
#6 — a staged slice with no image/mask in the segmentation manager is
     silently skipped, yet `_on_training_finished` marked EVERY staged
     label of a succeeded bucket as committed — the user believes a
     slice was trained on when it never was.
"""

from __future__ import annotations

import numpy as np
import tifffile

from packages.epicycle_training.training_manager import TrainingManager


class _SegManager:
    def __init__(self):
        self._inputs = {}
        self._masks = {}

    def set(self, label, img, mask):
        self._inputs[label] = img
        self._masks[label] = mask

    def get_input_stack(self, label):
        return self._inputs.get(label)

    def get_masks(self, label):
        return self._masks.get(label)


def test_missing_prior_pair_is_reported_to_the_ui(tmp_path):
    warnings = []
    mgr = TrainingManager()
    samples = mgr.prepare_round_dataset(
        segmentation_manager=_SegManager(),
        slice_labels=[],
        working_dir=str(tmp_path),
        round_id="round_00",
        prior_pairs=[{
            "img_uri": str(tmp_path / "gone_XY_z0001_img.tif"),
            "mask_uri": str(tmp_path / "gone_XY_z0001_mask.tif"),
            "view_group": "spatial",
        }],
        log_warn_cb=warnings.append,
    )
    assert samples == []
    assert warnings, "dropped prior pair must reach the UI log"
    assert "gone_XY_z0001_img.tif" in " ".join(warnings)


def test_skipped_staged_slice_is_reported_and_excluded(tmp_path):
    seg = _SegManager()
    good = "stk_XY_1"
    seg.set(good, np.random.rand(2, 8, 8).astype(np.float32),
            np.ones((8, 8), dtype=np.uint16))
    missing = "stk_XY_2"  # nothing registered -> silently skipped pre-fix

    warnings = []
    used = []
    mgr = TrainingManager()
    samples = mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=[good, missing],
        working_dir=str(tmp_path),
        round_id="round_00",
        log_warn_cb=warnings.append,
        used_slice_labels_out=used,
    )

    assert len(samples) == 1
    assert used == [good], "only the slice actually turned into a pair is 'used'"
    assert any(missing in w for w in warnings)


def test_used_labels_include_persisted_pairs(tmp_path):
    img = tmp_path / "a_img.tif"
    mask = tmp_path / "a_mask.tif"
    tifffile.imwrite(str(img), np.zeros((2, 4, 4), dtype=np.float32))
    tifffile.imwrite(str(mask), np.zeros((4, 4), dtype=np.int32))

    used = []
    mgr = TrainingManager()
    mgr.prepare_round_dataset(
        segmentation_manager=_SegManager(),
        slice_labels=["stk_XY_3"],
        working_dir=str(tmp_path),
        round_id="round_00",
        persisted_pairs={"stk_XY_3": {"img_uri": str(img), "mask_uri": str(mask)}},
        used_slice_labels_out=used,
    )
    assert used == ["stk_XY_3"]


def test_training_finished_leaves_unused_staged_slice_staged(tmp_path):
    """Nd-to-end: a staged label that produced no pair stays staged."""
    from .test_training_round_flow import _make_train_ctrl, _ListWidget

    ckpt = tmp_path / "models" / "round_00" / "epi_spatial_round_00"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("checkpoint", encoding="utf-8")

    def trainer_backend(**kwargs):
        return {
            "checkpoint_path": str(ckpt),
            "losses": [0.5],
            "mode": "custom_backend",
            "view_type": kwargs["view_type"],
        }

    ctrl = _make_train_ctrl(
        tmp_path,
        config={"model_name": "epi", "model_type": "cyto3", "epochs": 5},
        training_manager=TrainingManager(backend_trainer=trainer_backend),
    )
    ctrl.loaded_patch_name = "stk"
    good, ghost = "stk_XY_1", "stk_XY_9"
    ctrl.segmentation_manager.set(
        good,
        np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16),
    )
    ctrl._touch_sample(good, status="staged", view="XY")
    ctrl._touch_sample(ghost, status="staged", view="XY")  # no pixels anywhere
    ctrl.ui.data_to_add_list = _ListWidget([good, ghost])

    ctrl._train_model()

    assert ctrl.sample_records[good]["status"] == "committed"
    assert ctrl.sample_records[ghost]["status"] == "staged", (
        "a staged slice that was never turned into a training pair must not "
        "be marked committed"
    )
    text = chr(10).join(ctrl.ui.training_log_output.lines)
    assert ghost in text

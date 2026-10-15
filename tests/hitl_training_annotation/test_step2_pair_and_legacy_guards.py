"""Release-hardening #12, #13, #14, #15."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from packages.epicycle_io import HITLPersistenceService
from packages.epicycle_training.training_backend_adapter import (
    _report_min_mask_drops,
    _slice_label_for_sample,
    build_epicycle_training_backend,
)


# ---------------------------------------------------------------------------
# #12 — a legacy round_metadata.json without uris must not abort the restore
# ---------------------------------------------------------------------------

def test_legacy_round_metadata_without_uris_returns_none(tmp_path):
    svc = HITLPersistenceService()
    legacy_path = svc._legacy_round_metadata_path(str(tmp_path), "round_00")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    # A cherry-picked pair (schema 1.1) carries no img_uri / mask_uri —
    # `round_metadata_from_legacy` raises ValueError on it.
    legacy_path.write_text(
        json.dumps({"round_id": "round_00", "samples": [{"stack_id": "s"}]}),
        encoding="utf-8",
    )

    assert svc._load_round_metadata_with_compat(str(tmp_path), "round_00") is None


def test_legacy_round_metadata_corrupt_json_returns_none(tmp_path):
    svc = HITLPersistenceService()
    legacy_path = svc._legacy_round_metadata_path(str(tmp_path), "round_00")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("{not json", encoding="utf-8")

    assert svc._load_round_metadata_with_compat(str(tmp_path), "round_00") is None


# ---------------------------------------------------------------------------
# #13 — post-training the Model-source combo flips to "Round model"
# ---------------------------------------------------------------------------

class _Combo:
    def __init__(self):
        self.text = ""

    def setCurrentText(self, t):
        self.text = t


class _CheckBox:
    def __init__(self, checked=True):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


def _combo_ctrl(auto_update=True):
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)

    class _UI:
        pass

    ui = _UI()
    ui.training_log_output = _LogOutput()
    ui.model_source_combo = _Combo()
    ui.auto_update_model_cb = _CheckBox(auto_update)
    ctrl.ui = ui
    ctrl.active_curation_checkpoints = {"lateral": "", "axial": ""}
    ctrl.curation_model_source_mode = "auto"
    ctrl.current_config = {}
    ctrl._apply_curation_checkpoint_for_view = lambda **k: ""
    return ctrl


def test_promotion_flips_model_source_combo_to_round_model(tmp_path):
    ctrl = _combo_ctrl()
    ckpt = tmp_path / "weights"
    ckpt.write_text("w", encoding="utf-8")

    ctrl._promote_round_checkpoints_to_curation({"spatial": str(ckpt)})

    assert ctrl.curation_model_source_mode == "round"
    assert ctrl.ui.model_source_combo.text == "Round model"


def test_promotion_respects_auto_update_off(tmp_path):
    ctrl = _combo_ctrl(auto_update=False)
    ctrl.curation_model_source_mode = "imported"
    ckpt = tmp_path / "weights"
    ckpt.write_text("w", encoding="utf-8")

    ctrl._promote_round_checkpoints_to_curation({"spatial": str(ckpt)})

    assert ctrl.curation_model_source_mode == "imported"
    assert ctrl.ui.model_source_combo.text == ""


# ---------------------------------------------------------------------------
# #14 — image/mask shape mismatch must name the slice
# ---------------------------------------------------------------------------

def _pair(tmp_path, name, img_shape, mask_shape):
    img = tmp_path / f"{name}_img.tif"
    mask = tmp_path / f"{name}_mask.tif"
    tifffile.imwrite(str(img), np.zeros(img_shape, dtype=np.float32))
    tifffile.imwrite(str(mask), np.zeros(mask_shape, dtype=np.int32))
    return {
        "slice_label": name,
        "stack_id": "stk",
        "view": "XY",
        "slice_index": 1,
        "img_uri": str(img),
        "mask_uri": str(mask),
    }


def test_shape_mismatch_raises_naming_the_slice(tmp_path):
    backend = build_epicycle_training_backend()
    good = _pair(tmp_path, "stk_XY_1", (2, 16, 16), (16, 16))
    bad = _pair(tmp_path, "stk_XY_2", (2, 16, 16), (8, 8))

    with pytest.raises(ValueError) as exc:
        backend(
            round_id="round_00",
            config={"epochs": 1},
            working_dir=str(tmp_path),
            training_samples=[good, bad],
            view_type="spatial",
        )
    assert "stk_XY_2" in str(exc.value)
    assert "mismatch" in str(exc.value)


def test_slice_label_falls_back_to_identity_fields():
    assert _slice_label_for_sample(
        {"stack_id": "s", "view": "XZ", "slice_index": 7}
    ) == "s_XZ_7"


# ---------------------------------------------------------------------------
# #15 — cellpose's min-masks drop must be visible
# ---------------------------------------------------------------------------

def test_min_mask_drops_are_reported():
    warnings = []
    masks = [
        np.array([[0, 1], [2, 3]]),   # 3 objects
        np.array([[0, 0], [0, 1]]),   # 1 object -> dropped at threshold 2
    ]
    samples = [
        {"slice_label": "keep_XY_1"},
        {"slice_label": "drop_XY_2"},
    ]
    n = _report_min_mask_drops(
        masks=masks, samples=samples, min_train_masks=2,
        log_warn_cb=warnings.append,
    )
    assert n == 1
    assert warnings and "drop_XY_2" in warnings[0]
    assert "keep_XY_1" not in warnings[0]


def test_min_mask_drops_noop_when_threshold_zero():
    warnings = []
    n = _report_min_mask_drops(
        masks=[np.zeros((2, 2))], samples=[{"slice_label": "x"}],
        min_train_masks=0, log_warn_cb=warnings.append,
    )
    assert n == 0 and warnings == []


# ---------------------------------------------------------------------------
# Brick 1b #7 — no phantom val-loss file, all-drop fails fast, flag has a reader
# ---------------------------------------------------------------------------

def test_manifest_does_not_advertise_a_missing_val_loss_file():
    from packages.epicycle_training.step2_model_manifest import (
        build_step2_model_manifest,
    )

    m = build_step2_model_manifest(
        run_id="default", bucket="lateral", round_number=0,
        source_workspace="/ws", weights_filename="default_lateral_round_00",
        final_loss_train=0.3, final_loss_val=None,
    )
    assert m["loss_val_path"] is None
    assert m["loss_train_path"] == "default_lateral_round_00_train_loss.npy"

    m2 = build_step2_model_manifest(
        run_id="default", bucket="lateral", round_number=0,
        source_workspace="/ws", weights_filename="default_lateral_round_00",
        final_loss_train=0.3, final_loss_val=0.4,
    )
    assert m2["loss_val_path"] == "default_lateral_round_00_val_loss.npy"


def test_all_pairs_dropped_raises_before_cellpose():
    with pytest.raises(ValueError) as exc:
        _report_min_mask_drops(
            masks=[np.array([[0, 1]]), np.array([[0, 0]])],
            samples=[{"slice_label": "a_XY_1"}, {"slice_label": "b_XY_2"}],
            min_train_masks=5,
        )
    msg = str(exc.value)
    assert "every one" in msg and "min_train_masks=5" in msg
    assert "a_XY_1" in msg and "b_XY_2" in msg


def test_run_model_sources_flag_inference_failed(tmp_path):
    from napari_epicycle.hitl_training_annotation import _paths as p
    from napari_epicycle.hitl_training_annotation._run_management import (
        list_run_model_sources,
    )

    run_manifest = p.run_manifest_path(str(tmp_path), "default")
    run_manifest.parent.mkdir(parents=True, exist_ok=True)
    run_manifest.write_text(
        json.dumps({"run_id": "default", "display_name": "Default"}),
        encoding="utf-8",
    )

    from packages.epicycle_training.step2_model_manifest import (
        build_step2_model_manifest,
        write_step2_model_manifest,
    )

    for bucket, failed in (("lateral", True), ("axial", False)):
        md = p.model_dir(str(tmp_path), "default", bucket, 0)
        md.mkdir(parents=True, exist_ok=True)
        (md / f"default_{bucket}_round_00").write_text("w", encoding="utf-8")
        manifest = build_step2_model_manifest(
            run_id="default", bucket=bucket, round_number=0,
            source_workspace=str(tmp_path),
            weights_filename=f"default_{bucket}_round_00",
        )
        if failed:
            manifest["inference_failed"] = True
        write_step2_model_manifest(md / "manifest.json", manifest)

    by_bucket = {s["bucket"]: s for s in list_run_model_sources(str(tmp_path))}

    assert by_bucket["lateral"]["inference_failed"] is True
    assert "inference failed" in by_bucket["lateral"]["display_label"]
    # Identity string is untouched so other consumers keep comparing cleanly.
    assert by_bucket["lateral"]["run_display_name"] == "Default"
    assert by_bucket["axial"]["inference_failed"] is False
    assert by_bucket["axial"]["display_label"] == "Default"

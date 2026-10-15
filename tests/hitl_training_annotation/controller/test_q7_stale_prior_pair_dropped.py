"""Q7 (training side, owner ruling 2026-10-13 "drop the old ones"): a slice
trained in round k, reverted, re-edited and staged again has its OLD pair file
in round k's accepted dir.  The pairs handed to training must not include it --
the staged (newest) pair wins -- and the "pairs in all" display uses the same rule."""
from __future__ import annotations

from pathlib import Path

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController as C)
from tests.hitl_training_annotation.controller.test_step2_prior_round_filter import (
    _accepted_dir, _make_ctrl, _write_pair)


def _setup(tmp_path):
    new_img = _write_pair(_accepted_dir(tmp_path, 2), "P1", "XY", 1)  # current round
    records = {
        "P1_XY_z0001": {"status": "staged", "bucket": "lateral",
                        "img_uri": str(new_img),
                        "mask_uri": str(new_img).replace("_img", "_mask")},
        "P1_XY_z0002": {"status": "committed", "bucket": "lateral"},
    }
    ctrl = _make_ctrl(tmp_path, records, rounds=2)
    d = _accepted_dir(tmp_path, 0)
    _write_pair(d, "P1", "XY", 1)   # stale copy of z0001
    _write_pair(d, "P1", "XY", 2)
    for name in ("_prior_pair_total_excluding_staged", "_staged_slice_keys",
                 "_prior_pairs_for_training"):
        setattr(ctrl, name, getattr(C, name).__get__(ctrl))
    return ctrl


def _names(pairs):
    return sorted(Path(p["img_uri"]).name for p in pairs)


def test_stale_pair_of_restaged_slice_is_dropped_from_training(tmp_path):
    ctrl = _setup(tmp_path)
    assert len(ctrl._collect_prior_round_pairs("lateral")) == 2  # raw scan keeps both
    pairs = ctrl._prior_pairs_for_training(["lateral"])
    assert _names(pairs) == ["P1_XY_z0002_img.tif"]
    # prior (z0002) + staged (z0001, current round's file) -> z0001 once
    staged = [r for r in ctrl.sample_records.values() if r["status"] == "staged"]
    assert len(pairs) + len(staged) == 2


def test_display_total_equals_trained_prior_count(tmp_path):
    ctrl = _setup(tmp_path)
    assert ctrl._prior_pair_total_excluding_staged("lateral") == len(
        ctrl._prior_pairs_for_training(["lateral"]))


def test_committed_only_slice_still_trains(tmp_path):
    ctrl = _setup(tmp_path)
    ctrl.sample_records["P1_XY_z0001"]["status"] = "committed"
    assert len(ctrl._prior_pairs_for_training(["lateral"])) == 2


def test_other_bucket_staged_record_does_not_drop(tmp_path):
    ctrl = _setup(tmp_path)
    ctrl.sample_records["P1_XY_z0001"]["bucket"] = "axial"
    assert len(ctrl._prior_pairs_for_training(["lateral"])) == 2


def test_train_model_uses_the_filtered_collector():
    import inspect
    src = inspect.getsource(C._train_model)
    assert "_prior_pairs_for_training(" in src

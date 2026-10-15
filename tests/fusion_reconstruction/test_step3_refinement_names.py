"""Model Refinement Brick A, M12 / I1: numbered auto-names."""

from __future__ import annotations

import json

from napari_epicycle.fusion_reconstruction import _paths
from tests.fusion_reconstruction._refine_launch_harness import (
    make_launch_controller,
)


def _mk(tmp_path, *names):
    for n in names:
        _paths.refined_model_dir(tmp_path, n).mkdir(parents=True)


def _launch(tmp_path, monkeypatch, **kw):
    ctrl, h = make_launch_controller(tmp_path, monkeypatch, **kw)
    ctrl._append_log = lambda _m: None
    ctrl._launch_refinement()
    return ctrl, h


def test_first_run_is_refined_01(tmp_path, monkeypatch):
    ctrl, h = _launch(tmp_path, monkeypatch)
    assert {w.bundle_name for w in h.workers} == {"refined_01"}


def test_next_free_number_follows_the_highest(tmp_path, monkeypatch):
    _mk(tmp_path, "refined_01", "refined_02")
    ctrl, h = _launch(tmp_path, monkeypatch)
    assert h.workers[0].bundle_name == "refined_03"


def test_refined_09_becomes_refined_10(tmp_path, monkeypatch):
    _mk(tmp_path, "refined_09")
    ctrl, h = _launch(tmp_path, monkeypatch)
    assert h.workers[0].bundle_name == "refined_10"


def test_other_folders_are_ignored(tmp_path, monkeypatch):
    _mk(tmp_path, "refined_20261005_120000", "wound_model", "refined_x")
    ctrl, h = _launch(tmp_path, monkeypatch)
    assert h.workers[0].bundle_name == "refined_01"


def test_typed_name_is_kept_as_typed(tmp_path, monkeypatch):
    _mk(tmp_path, "refined_05")
    ctrl, h = _launch(tmp_path, monkeypatch, name="Wound Domain v2")
    assert h.workers[0].bundle_name == "Wound Domain v2"


def test_both_passes_share_the_numbered_name(tmp_path, monkeypatch):
    ctrl, h = _launch(tmp_path, monkeypatch)
    ctrl._launch_next_refinement()
    assert [w.bundle_name for w in h.workers] == ["refined_01", "refined_01"]


def test_typed_name_collision_is_still_asked(tmp_path, monkeypatch):
    _mk(tmp_path, "taken")
    mp = _paths.refined_model_manifest_path(tmp_path, "taken")
    mp.write_text(
        json.dumps({"lateral": {"x": 1}, "axial": {"x": 1}}), encoding="utf-8",
    )
    ctrl, h = make_launch_controller(tmp_path, monkeypatch, name="taken")
    ctrl._append_log = lambda _m: None
    asked = []
    ctrl._show_refinement_collision_dialog = (
        lambda n, c: asked.append(n) or "cancel"
    )
    ctrl._launch_refinement()
    assert asked == ["taken"]
    assert h.workers == []
    assert ctrl._refinement_bundle_name is None
    assert ctrl._refinement_launch_snapshot is None


def test_next_refined_name_with_no_models_folder(tmp_path, monkeypatch):
    ctrl, _h = make_launch_controller(tmp_path, monkeypatch)
    assert ctrl._next_refined_name() == "refined_01"

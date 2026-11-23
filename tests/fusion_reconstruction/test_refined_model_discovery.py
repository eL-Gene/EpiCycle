"""Model discovery never picks a mid-training checkpoint: a file under a
refined bundle's own ``models`` sub-folder is skipped, and so is a bundle
that has such a folder but no ``manifest.json``.  Older layouts keep
working."""
from __future__ import annotations

from pathlib import Path

from .test_model_reader_unify import (
    _make_controller,
    _plant_legacy_workspace_models,
    _plant_step3_refined,
)


def _bundle(ws: Path, name: str) -> Path:
    return ws / "artifacts" / "step3" / "refinement" / "models" / name


def test_a_both_mode_bundle_does_not_yield_its_models_subfolder(tmp_path):
    root = _bundle(tmp_path, "refined_01")
    (root / "models").mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (root / "lateral").write_bytes(b"final lateral")
    (root / "models" / "axial").write_bytes(b"checkpoint")
    ctrl = _make_controller(tmp_path)
    ctrl._discover_workspace_models()
    assert ctrl.ui.lateral_model_path.text().endswith("lateral")
    assert ctrl.ui.axial_model_path.text() == ""


def test_a_bundle_with_only_models_and_no_manifest_is_not_picked(tmp_path):
    root = _bundle(tmp_path, "refined_02")
    (root / "models").mkdir(parents=True)
    (root / "models" / "lateral").write_bytes(b"checkpoint")
    ctrl = _make_controller(tmp_path)
    ctrl._discover_workspace_models()
    assert ctrl.ui.lateral_model_path.text() == ""


def test_a_bundle_without_a_manifest_and_without_models_still_works(tmp_path):
    f = _plant_step3_refined(tmp_path, "refined_03", "axial")
    ctrl = _make_controller(tmp_path)
    ctrl._discover_workspace_models()
    assert ctrl.ui.axial_model_path.text() == str(f)


def test_a_legacy_models_round_folder_is_still_picked(tmp_path):
    f = _plant_legacy_workspace_models(tmp_path, "round_1", "lateral")
    ctrl = _make_controller(tmp_path)
    ctrl._discover_workspace_models()
    assert ctrl.ui.lateral_model_path.text() == str(f)

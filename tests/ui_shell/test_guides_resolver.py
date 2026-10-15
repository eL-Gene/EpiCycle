"""The guide folder resolver: installed copy first, repo docs/guides second."""
from pathlib import Path

import pytest

MODULES = [
    ("napari_epicycle.annotator.advanced_features_widget", "_GUIDES_DIR"),
    ("napari_epicycle.annotator.curation_2d_widget", "_ANNOTATOR_GUIDES_DIR"),
    ("napari_epicycle.annotator.curation_3d_widget", "_ANNOTATOR_GUIDES_DIR"),
    ("napari_epicycle.data_preparation.ui", "_GUIDES_DIR"),
    ("napari_epicycle.fusion_reconstruction.ui", "_GUIDES_DIR"),
    ("napari_epicycle.hitl_training_annotation.ui.main_widget", "_STEP2_GUIDES_DIR"),
    ("napari_epicycle.hitl_training_annotation.ui._step2_curation", "_GUIDES_DIR"),
]


def test_dev_tree_uses_repo_docs_guides():
    from napari_epicycle.ui_shell import _guides
    root = Path(_guides.__file__).resolve().parents[2]
    assert _guides.guides_dir() == root / "docs" / "guides"
    assert _guides.guide_path("step3_session.md") == root / "docs" / "guides" / "step3_session.md"


def test_installed_copy_wins_when_present(tmp_path, monkeypatch):
    from napari_epicycle.ui_shell import _guides
    pkg = tmp_path / "napari_epicycle"
    (pkg / "ui_shell").mkdir(parents=True)
    (pkg / "guides").mkdir()
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    monkeypatch.setattr(_guides, "__file__", str(pkg / "ui_shell" / "_guides.py"))
    assert _guides.guides_dir() == pkg / "guides"


def test_falls_back_to_repo_docs(tmp_path, monkeypatch):
    from napari_epicycle.ui_shell import _guides
    pkg = tmp_path / "napari_epicycle"
    (pkg / "ui_shell").mkdir(parents=True)
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    monkeypatch.setattr(_guides, "__file__", str(pkg / "ui_shell" / "_guides.py"))
    assert _guides.guides_dir() == tmp_path / "docs" / "guides"


@pytest.mark.parametrize("mod,name", MODULES)
def test_modules_share_the_resolver(mod, name):
    import importlib
    from napari_epicycle.ui_shell._guides import guides_dir
    assert Path(getattr(importlib.import_module(mod), name)) == guides_dir()


def test_segmentation_guide_path_uses_resolver():
    from napari_epicycle.segmentation.ui import _guide_path
    from napari_epicycle.ui_shell._guides import guide_path
    assert _guide_path("step4_session.md") == str(guide_path("step4_session.md"))

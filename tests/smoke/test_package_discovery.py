"""Every source directory must be a regular package.

A directory without an ``__init__.py`` is skipped by ``find_packages`` and
never reaches the wheel, so a non-editable install fails at launch with
``ModuleNotFoundError``.  Editable installs hide the problem because they
import straight from the source tree.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("napari_epicycle", "packages")
PATTERNS = ["napari_epicycle", "napari_epicycle.*", "packages", "packages.*"]


def test_every_source_directory_has_init():
    offenders = []
    for root in ROOTS:
        for d in sorted((REPO / root).rglob("*")):
            if not d.is_dir() or "__pycache__" in d.parts:
                continue
            has_py = any(p.suffix == ".py" for p in d.iterdir() if p.is_file())
            if has_py and not (d / "__init__.py").exists():
                offenders.append(str(d.relative_to(REPO)))
    assert not offenders, "directories with .py files but no __init__.py: " + ", ".join(offenders)


def test_find_packages_sees_step2_and_utils():
    from setuptools import find_packages

    found = set(find_packages(where=str(REPO), include=PATTERNS))
    for name in (
        "napari_epicycle.hitl_training_annotation",
        "napari_epicycle.hitl_training_annotation.controller",
        "napari_epicycle.hitl_training_annotation.ui",
        "napari_epicycle.utils",
    ):
        assert name in found, f"{name} missing from find_packages result"

"""Where the help guides live.

Installed wheel: ``napari_epicycle/guides/`` (copied from ``docs/guides/`` at
build time by ``setup.py``).  Repo or editable install: ``<repo>/docs/guides/``.
"""
from pathlib import Path


def guides_dir() -> Path:
    pkg = Path(__file__).resolve().parents[1]
    installed = pkg / "guides"
    if installed.is_dir():
        return installed
    return pkg.parent / "docs" / "guides"


def guide_path(name: str) -> Path:
    return guides_dir() / name

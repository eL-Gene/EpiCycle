"""A built wheel carries the manifest, icons and every guide (no clone needed).

The build runs on a COPY of the files a clean clone has (not on the live tree):
pip builds in-tree, and the live tree's ``build/`` persists between runs, so a
wheel built there can carry guides from an older build and pass even with the
build_py hook broken -- and it leaves ``build/`` + ``*.egg-info`` behind.
"""
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: what a clean clone needs to build (the rest of the repo is not packaged)
_FILES = ("setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in",
          "README.md", "LICENSE", "requirements-lock.txt")
_DIRS = ("napari_epicycle", "packages", "docs/guides")


def _clean_copy(dst: Path) -> Path:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    dst.mkdir()
    for name in _FILES:
        shutil.copy2(ROOT / name, dst / name)
    for name in _DIRS:
        shutil.copytree(ROOT / name, dst / name, ignore=ignore)
    # a stray dev-tree copy must not stand in for the build hook's copy
    shutil.rmtree(dst / "napari_epicycle" / "guides", ignore_errors=True)
    return dst


@pytest.mark.slow
def test_wheel_contains_manifest_icons_and_guides(tmp_path):
    pytest.importorskip("setuptools")
    src = _clean_copy(tmp_path / "s")  # short: Windows path length
    guides = sorted(p.name for p in (ROOT / "docs" / "guides").glob("*.md"))
    assert guides

    out = tmp_path / "wheel"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "-w", str(out), str(src)],
        check=True, capture_output=True, cwd=str(tmp_path),
    )
    whl = next(out.glob("napari_epicycle-*.whl"))
    names = set(zipfile.ZipFile(whl).namelist())
    assert "napari_epicycle/napari.yaml" in names
    assert any(n.endswith(".svg") for n in names)
    for md in guides:
        assert f"napari_epicycle/guides/{md}" in names, md

    # a wheel built FROM the sdist (python -m build) needs the guides in it
    sd = tmp_path / "sdist"
    subprocess.run(
        [sys.executable, "setup.py", "-q", "sdist", "-d", str(sd)],
        check=True, capture_output=True, cwd=str(src),
    )
    with tarfile.open(next(sd.glob("napari_epicycle-*.tar.gz"))) as tar:
        members = {m.split("/", 1)[1] for m in tar.getnames() if "/" in m}
    for md in guides:
        assert f"docs/guides/{md}" in members, md

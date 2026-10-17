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


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    """Build the wheel once for the whole file; return (source copy, wheel path)."""
    pytest.importorskip("setuptools")
    tmp_path = tmp_path_factory.mktemp("whl")
    src = _clean_copy(tmp_path / "s")  # short: Windows path length
    out = tmp_path / "wheel"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "-w", str(out), str(src)],
        check=True, capture_output=True, cwd=str(tmp_path),
    )
    return src, next(out.glob("napari_epicycle-*.whl"))


@pytest.mark.slow
def test_wheel_contains_manifest_icons_and_guides(built_wheel, tmp_path):
    src, whl = built_wheel
    guides = sorted(p.name for p in (ROOT / "docs" / "guides").glob("*.md"))
    assert guides

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


@pytest.mark.slow
def test_wheel_contains_every_tracked_source_file(built_wheel):
    """Every file git tracks under napari_epicycle/ and packages/ is in the wheel.

    A folder without an ``__init__.py`` is silently left out of a regular
    install (an editable install hides it, because it reads the source tree),
    so a missing file here means a non-editable install would be broken.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    probe = subprocess.run(
        ["git", "ls-files", "napari_epicycle", "packages"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if probe.returncode != 0:
        pytest.skip("not a git checkout: " + probe.stderr.strip())
    tracked = [
        line for line in probe.stdout.splitlines()
        if line and "__pycache__" not in line and not line.endswith(".pyc")
    ]
    if not tracked:
        pytest.skip("git lists no tracked files here")

    _, whl = built_wheel
    names = set(zipfile.ZipFile(whl).namelist())
    missing = sorted(p for p in tracked if p not in names)
    listing = chr(10).join("  " + m for m in missing)
    assert not missing, (
        f"{len(missing)} tracked file(s) are not in the built wheel:" + chr(10) + listing
    )

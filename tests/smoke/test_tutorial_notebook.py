"""Static checks on the three notebooks in tutorials/segmentation/ (no cell is executed)."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "tutorials" / "segmentation"
NOTEBOOK_NAMES = (
    "01_run_whole_tissue_segmentation.ipynb",
    "02_reconcile_cells_and_nuclei.ipynb",
    "03_cleanup_to_consensus.ipynb",
)
NOTEBOOK = NOTEBOOK_DIR / NOTEBOOK_NAMES[0]

_ABS_PATH = re.compile(r"[A-Za-z]:\\|/home/|/Users/")
_OLD_IMPORT = re.compile(r"^\s*(from|import)\s+epicycle(\.|\s|$)", re.M)
_PKG_MODULE = re.compile(
    r"^\s*(?:from|import)\s+(packages\.epicycle_scaledseg[\w.]*)", re.M
)


@pytest.fixture(scope="module", params=NOTEBOOK_NAMES)
def nb(request):
    return json.loads((NOTEBOOK_DIR / request.param).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb01():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell) -> str:
    return "".join(cell["source"])


def test_has_cells_and_valid_format(nb):
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10


def test_no_stored_outputs(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_no_old_engine_imports(nb):
    for cell in nb["cells"]:
        assert not _OLD_IMPORT.search(_src(cell)), _src(cell)[:80]


def test_no_absolute_paths(nb):
    for cell in nb["cells"]:
        m = _ABS_PATH.search(_src(cell))
        assert m is None, f"absolute path {m.group(0)!r} in: {_src(cell)[:80]}"


def test_honest_defaults_present(nb01):
    nb = nb01
    code = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'MODE = "full"' in code
    assert 'FEAT_FEATURE_SET = "epicycle"' in code
    assert "RUN_FEATURES = True" in code
    assert "feature_set=FEAT_FEATURE_SET" in code
    assert "VOXEL_SIZE_UM = None" in code and "VOXEL_SIZE_UM is None" in code


def test_imported_modules_exist(nb):
    modules = set()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            modules.update(_PKG_MODULE.findall(_src(cell)))
    assert len(modules) >= 1
    for name in sorted(modules):
        importlib.import_module(name)

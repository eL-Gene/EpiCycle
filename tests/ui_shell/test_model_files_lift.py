"""R6a's lift: ``file_fingerprint``, ``_training_config_files`` and
``read_trained_channels`` moved VERBATIM from ``segmentation/_step4_models``
into the Qt-free ``ui_shell/_model_files``, Step 4 re-exporting the same objects.

* the bodies are byte-identical to the pre-lift source (``9bf02891``): a
  SHA-256 of each function's source as it stood there, and -- when git can
  read that commit -- the exact text;
* ``_step4_models`` answers with the same objects;
* the new module imports no step package and no Qt (AST).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import subprocess
from pathlib import Path

import pytest

import napari_epicycle  # noqa: F401  (the app's import order)
from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.ui_shell import _model_files as MF

REPO = Path(__file__).resolve().parents[2]
PRE_LIFT = "9bf02891"
OLD_FILE = "napari_epicycle/segmentation/_step4_models.py"

#: SHA-256 of each function's source at ``9bf02891`` (LF line ends, no
#: trailing newline) -- ``ast.get_source_segment`` of the def.
PRE_LIFT_SHA256 = {
    "file_fingerprint":
        "83dc6d56999e75eb50816a44fe5c7f973bdc4a0edc86477e2e614d5215530637",
    "read_trained_channels":
        "89d2847ff2853469d01e1968e3f5d31899ded78dd960dadf8d1b99eff4363709",
}


#: Step 3 brick B audit (M1, M2): ``_training_config_files`` is NO LONGER
#: verbatim -- the file named for the weights comes first, and the bucket
#: names / glob fallback only for a folder with one weights file.  It is
#: still lifted (one object, Step 4 re-exports it); its behaviour is held by
#: ``test_step3_pr_parity_b.py`` (M1, M2) and Step 4's reader tests.
LIFTED = sorted(set(PRE_LIFT_SHA256) | {"_training_config_files"})


def _source(fn) -> str:
    return inspect.getsource(fn).replace("\r\n", "\n").rstrip("\n")


@pytest.mark.parametrize("name", sorted(PRE_LIFT_SHA256))
def test_the_body_is_the_pre_lift_body(name):
    got = hashlib.sha256(_source(getattr(MF, name)).encode()).hexdigest()
    assert got == PRE_LIFT_SHA256[name]


@pytest.mark.parametrize("name", sorted(PRE_LIFT_SHA256))
def test_the_body_diffs_clean_against_the_pre_lift_commit(name):
    try:
        old = subprocess.run(
            ["git", "show", f"{PRE_LIFT}:{OLD_FILE}"], cwd=str(REPO),
            capture_output=True, text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    if old.returncode != 0:
        pytest.skip(f"{PRE_LIFT} is not readable here")
    src = old.stdout.replace("\r\n", "\n")
    tree = ast.parse(src)
    defs = {n.name: ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef)}
    assert defs[name] == _source(getattr(MF, name))


@pytest.mark.parametrize("name", LIFTED)
def test_step4_answers_with_the_same_object(name):
    assert getattr(M, name) is getattr(MF, name)
    assert getattr(MF, name).__module__ == "napari_epicycle.ui_shell._model_files"


def test_step4_no_longer_defines_them():
    tree = ast.parse((REPO / OLD_FILE).read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert not (defined & set(LIFTED)), defined & set(LIFTED)


def test_the_module_imports_no_step_and_no_qt():
    from tests.ui_shell.test_recon_shared_modules_imports import (
        SHARED, _step_imports,
    )
    assert "_model_files.py" in SHARED
    tree = ast.parse((REPO / "napari_epicycle" / "ui_shell" /
                      "_model_files.py").read_text(encoding="utf-8"))
    assert not _step_imports(tree)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert not any(m.startswith(("qtpy", "PyQt", "napari", "superqt"))
                   for m in mods), mods


def test_the_lifted_readers_still_read(tmp_path):
    w = tmp_path / "bundle" / "lateral"
    w.parent.mkdir()
    w.write_bytes(b"x" * 7)
    (w.parent / "lateral_training_config.json").write_text(
        '{"hyperparameters": {"channels": [2, 1]}}', encoding="utf-8")
    assert MF.file_fingerprint(w)["size"] == 7
    assert MF.file_fingerprint(tmp_path / "gone") is None
    assert MF.read_trained_channels(w) == [2, 1]
    assert MF.read_trained_channels("") is None
    assert w.parent / "lateral_training_config.json" in \
        MF._training_config_files(w)

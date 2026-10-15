"""The modules Steps 3 and 4 share import nothing from a step package

``ui_shell`` is below the steps: a shared module that imported
``segmentation._step4_text`` would land every Step 4 rewording in Step 3
silently, so the delegate's and the dialog's Step 4 inputs
arrive as constructor keywords instead.  AST-only.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SHELL = Path(__file__).resolve().parents[2] / "napari_epicycle" / "ui_shell"
SHARED = ["_recon_settings_text.py", "_recon_presets.py",
          "_recon_settings_widgets.py", "_stack_row_delegate.py",
          "_stack_thumbs.py", "_model_files.py",
          # R7a: Set spacing… shared with Step 3, words via ``texts=``
          "_set_spacing_dialog.py"]
STEP_PACKAGES = ("segmentation", "fusion_reconstruction", "data_preparation",
                 "hitl_training_annotation", "tiled_segmentation",
                 "model_selection", "annotator")


_IMPORT_CALLS = ("import_module", "__import__")


def _names_a_step(dotted: str) -> bool:
    return any(p in STEP_PACKAGES for p in dotted.split("."))


def _call_name(func) -> str:
    """``importlib.import_module`` / ``import_module`` / ``__import__``."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _step_imports(tree) -> list:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            names = [a.name for a in node.names]
            # "from .. import segmentation" and "from napari_epicycle import
            # segmentation" name the package in the list
            from_root = node.level >= 2 or (
                node.level == 0 and parts[-1] == "napari_epicycle")
            if any(p in STEP_PACKAGES for p in parts) or (
                    from_root and any(n.split(".")[0] in STEP_PACKAGES
                                      for n in names)):
                out.append(("." * node.level) + mod + ":" + ",".join(names))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if _names_a_step(a.name):
                    out.append(a.name)
        elif (isinstance(node, ast.Call)
              and _call_name(node.func) in _IMPORT_CALLS):
            # importlib.import_module("napari_epicycle.segmentation...") and
            # __import__("...") in their string forms
            for arg in node.args[:1]:
                if (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and _names_a_step(arg.value)):
                    out.append(_call_name(node.func) + ":" + arg.value)
    return out


@pytest.mark.parametrize("module", SHARED)
def test_a_shared_module_imports_no_step_package(module):
    tree = ast.parse((SHELL / module).read_text(encoding="utf-8"))
    assert not _step_imports(tree), _step_imports(tree)


def test_the_text_and_preset_tables_are_qt_free():
    for module in ("_recon_settings_text.py", "_recon_presets.py",
                   "_model_files.py"):
        tree = ast.parse((SHELL / module).read_text(encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add(node.module or "")
        assert not any(m.startswith(("qtpy", "PyQt", "napari", "superqt"))
                       for m in mods), (module, mods)


def test_the_check_has_teeth():
    bad = ast.parse("from ..segmentation import _step4_text as T\n"
                    "from napari_epicycle.fusion_reconstruction import ui\n"
                    "import napari_epicycle.annotator.launcher\n"
                    "from .. import data_preparation\n")
    assert len(_step_imports(bad)) == 4
    good = ast.parse("from .theme import STEP4_ACCENT\n"
                     "from ._recon_settings_text import SEG_JOIN_DESC\n"
                     "from qtpy.QtCore import Qt\n")
    assert not _step_imports(good)


_NEW_FORMS = ("from napari_epicycle import {pkg}\n",
              "from napari_epicycle import ui_shell, {pkg} as S\n",
              "import importlib\n"
              "importlib.import_module('napari_epicycle.{pkg}.ui')\n",
              "from importlib import import_module\n"
              "import_module('napari_epicycle.{pkg}')\n",
              "__import__('napari_epicycle.{pkg}.controller')\n")


@pytest.mark.parametrize("form", range(len(_NEW_FORMS)))
@pytest.mark.parametrize("pkg", STEP_PACKAGES)
def test_the_check_catches_the_root_and_string_forms(pkg, form):
    """A level-0 ``from napari_epicycle import <step>`` and the string forms
    ``importlib.import_module`` / ``import_module`` / ``__import__``."""
    bad = ast.parse(_NEW_FORMS[form].format(pkg=pkg))
    assert len(_step_imports(bad)) == 1, _step_imports(bad)


def test_the_new_forms_leave_shared_imports_alone():
    good = ast.parse("from napari_epicycle import ui_shell\n"
                     "from napari_epicycle.ui_shell import theme\n"
                     "import importlib\n"
                     "importlib.import_module("
                     "'napari_epicycle.ui_shell.theme')\n"
                     "__import__('qtpy')\n"
                     "from . import theme\n")
    assert not _step_imports(good), _step_imports(good)

"""I-12: no lambda or ``functools.partial`` is connected to a signal of
a Qt-CREATED child (``selectionModel()``, ``viewport()``, a header, a
scroll bar) in the Step 4 modules, and the B1a modules connect no lambda at
all.  AST-only."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SEG = Path(__file__).resolve().parents[2] / "napari_epicycle" / "segmentation"
QT_CREATED = {"selectionModel", "viewport", "horizontalHeader",
              "verticalHeader", "verticalScrollBar", "horizontalScrollBar",
              "view", "model"}
#: the modules Step 4 moved into ui_shell/
#: are read where they live now, never through their alias
MOVED = ["ui_shell/_stack_row_delegate.py",
         "ui_shell/_recon_settings_widgets.py",
         "ui_shell/_stack_thumbs.py"]
#: the shared Set spacing dialog Step 4's module SUBCLASSES since R7a
#:: checked where its code lives (it is
#: not an alias, so it is not in ``MOVED``)
LIFTED = ["ui_shell/_set_spacing_dialog.py"]
MODULES = ["ui.py", "controller.py", "_set_spacing_dialog.py"] + MOVED + LIFTED
B1A_ONLY = ["_set_spacing_dialog.py"] + MOVED + LIFTED


def _path(module: str) -> Path:
    return (SEG.parent / module) if "/" in module else (SEG / module)


def _is_lambda_or_partial(node) -> bool:
    if isinstance(node, ast.Lambda):
        return True
    if isinstance(node, ast.Call):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        return name == "partial"
    return False


def _chain_has_qt_child(node) -> bool:
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr in QT_CREATED):
            return True
    return False


def _connects(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect" and node.args):
            yield node


@pytest.mark.parametrize("module", MODULES)
def test_no_lambda_on_a_qt_created_child(module):
    tree = ast.parse(_path(module).read_text(encoding="utf-8"))
    bad = [ast.unparse(n)[:100] for n in _connects(tree)
           if _is_lambda_or_partial(n.args[0])
           and _chain_has_qt_child(n.func.value)]
    assert not bad, bad


@pytest.mark.parametrize("module", B1A_ONLY)
def test_the_b1a_modules_connect_no_lambda(module):
    tree = ast.parse(_path(module).read_text(encoding="utf-8"))
    bad = [ast.unparse(n)[:100] for n in _connects(tree)
           if _is_lambda_or_partial(n.args[0])]
    assert not bad, bad


def test_the_check_has_teeth():
    tree = ast.parse("lw.selectionModel().selectionChanged.connect("
                     "lambda *a: self.f())\n")
    n = next(_connects(tree))
    assert _is_lambda_or_partial(n.args[0])
    assert _chain_has_qt_child(n.func.value)


def test_the_moved_modules_are_read_where_they_live():
    """The moved modules are checked
    in ``ui_shell/`` -- the old ``segmentation/`` path is a three-line alias
    an AST check would pass vacuously."""
    for module in MOVED:
        real = _path(module)
        assert real.parent.name == "ui_shell", module
        assert len(real.read_text(encoding="utf-8").splitlines()) > 100
        old = SEG / real.name
        if old.exists():
            assert "sys.modules[__name__]" in old.read_text(encoding="utf-8")

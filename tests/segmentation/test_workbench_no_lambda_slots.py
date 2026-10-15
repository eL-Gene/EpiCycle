"""I-12 ("Signals and the garbage
collector"): no lambda or ``functools.partial`` is connected to a signal in
the Workbench modules.  A slot held only by PyQt on a Qt-created child's
signal can have its closure cleared by napari's per-layer-removal
``gc.collect()``; the Workbench connects bound methods (or signals) only,
which is stricter than the rule and easy to hold."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "napari_epicycle" / "segmentation"
FILES = ("analysis_widget.py", "_workbench_text.py")


def _is_partial(node) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "partial") or (
        isinstance(f, ast.Attribute) and f.attr == "partial")


@pytest.mark.parametrize("name", FILES)
def test_no_lambda_or_partial_is_connected(name):
    tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"):
            for arg in node.args:
                if isinstance(arg, ast.Lambda) or _is_partial(arg):
                    bad.append(node.lineno)
    assert not bad, f"{name}: lambda/partial connected at lines {bad}"


def test_the_check_catches_a_lambda():
    tree = ast.parse("x.sig.connect(lambda: 1)\ny.s.connect(partial(f, 1))")
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "connect"
            and any(isinstance(a, ast.Lambda) or _is_partial(a)
                    for a in n.args)]
    assert len(hits) == 2

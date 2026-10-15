"""No string literal, f-string,
``"..." + x``, ``a if c else "b"`` or ``"...".format(...)`` reaches a
``setText`` / ``setToolTip`` / ``setWindowTitle`` / ``setPlaceholderText``
or a status push in ``_qc_plot_area.py`` -- every word on screen comes from
``_qc_plot_text``.  AST-only.  An EMPTY string (a clear) is allowed.
"""

from __future__ import annotations

import ast
from pathlib import Path

AREA = (Path(__file__).resolve().parents[2] / "napari_epicycle"
        / "fusion_reconstruction" / "_qc_plot_area.py")

TEXT_CALLS = {"setText", "setToolTip", "setWindowTitle", "setPlaceholderText"}
#: the area's own status / message pushes (the text is the FIRST argument)
PUSHES = {"set_status", "set_card_message", "set_message"}


def _bad(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip() != ""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp):
        return _bad(node.left) or _bad(node.right)
    if isinstance(node, ast.IfExp):
        return _bad(node.body) or _bad(node.orelse)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)):
        return True
    if isinstance(node, ast.BoolOp):
        return any(_bad(v) for v in node.values)
    return False


def offences(source: str) -> list:
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (
            f.id if isinstance(f, ast.Name) else "")
        if name in TEXT_CALLS | PUSHES and node.args:
            if _bad(node.args[0]):
                out.append(f"line {node.lineno}: {name}({ast.unparse(node.args[0])[:60]})")
        # a QWidget built with its text: QPushButton("..."), QCheckBox("...")
        if name in ("QPushButton", "QCheckBox", "QLabel") and node.args:
            if _bad(node.args[0]):
                out.append(f"line {node.lineno}: {name}({ast.unparse(node.args[0])[:60]})")
    return out


def test_no_literal_reaches_the_screen():
    bad = offences(AREA.read_text(encoding="utf-8"))
    assert bad == [], "\n".join(bad)


def test_the_check_has_teeth():
    planted = [
        'w.setText("Hello")',
        'w.setToolTip(f"{n} labels")',
        'w.setText("a" + x)',
        'w.setText(x if c else "b")',
        'w.setText("{n}".format(n=1))',
        'pw.set_status("Careful", "warn")',
        'QPushButton("Go")',
        'w.setText(x or "fallback")',
    ]
    for line in planted:
        assert offences(line), line
    for ok in ('w.setText("")', "w.setText(T.SEE_NO_PATCH)",
               "w.setText(T.SEE_LABELS.format(n=3))",
               'pw.set_status("", "info")'):
        assert not offences(ok), ok

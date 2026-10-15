"""18: every sentence the Workbench shows is a
named constant of ``_workbench_text.py``.

An AST walk of ``analysis_widget.py`` refuses any string literal (other
than the empty string) or f-string that reaches ``setText`` on one of the
Workbench's status labels (``AnalysisControlsWidget.STATUS_LABELS``), and
any literal help text reaching ``setToolTip`` on a Workbench control.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AW = ROOT / "napari_epicycle" / "segmentation" / "analysis_widget.py"


def _status_labels() -> set:
    from napari_epicycle.segmentation.analysis_widget import AnalysisControlsWidget

    return set(AnalysisControlsWidget.STATUS_LABELS)


def _is_literal(node) -> bool:
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value != ""
    if isinstance(node, ast.BinOp):
        return _is_literal(node.left) or _is_literal(node.right)
    if isinstance(node, ast.IfExp):
        return _is_literal(node.body) or _is_literal(node.orelse)
    return False


def _calls(method: str):
    tree = ast.parse(AW.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == method and node.args):
            yield node


def _target_name(func: ast.Attribute):
    """``self.<name>.setText`` -> ``<name>``."""
    val = func.value
    if isinstance(val, ast.Attribute):
        return val.attr
    return None


def test_the_status_label_set_is_real():
    names = _status_labels()
    src = AW.read_text(encoding="utf-8")
    for n in names:
        assert f"self.{n} = " in src, n


def test_no_literal_reaches_a_status_label():
    labels = _status_labels()
    bad = []
    for call in _calls("setText"):
        name = _target_name(call.func)
        if name in labels and _is_literal(call.args[0]):
            bad.append(f"line {call.lineno}: {name}")
    assert not bad, bad


def test_no_literal_sentence_in_a_status_constructor():
    """``QLabel("...", parent)`` for a status label must be a constant."""
    labels = _status_labels()
    tree = ast.parse(AW.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Attribute) and tgt.attr in labels):
            continue
        call = node.value
        if isinstance(call, ast.Call) and call.args and _is_literal(call.args[0]):
            bad.append(f"line {node.lineno}: {tgt.attr}")
    assert not bad, bad


def test_no_literal_help_text():
    """Tooltips come from WORKBENCH_HELP_LINES (``H[...]``) or a constant."""
    bad = []
    for call in _calls("setToolTip"):
        if _is_literal(call.args[0]):
            bad.append(f"line {call.lineno}")
    # One pre-existing tooltip of the feature table shows the skew VALUE
    # ("Fisher's skewness: 1.234"); it is data, not a sentence.
    allowed = _allowed_data_tooltips()
    assert set(bad) <= allowed, sorted(set(bad) - allowed)
    assert len(allowed) <= 1


def _allowed_data_tooltips() -> set:
    src = AW.read_text(encoding="utf-8").splitlines()
    return {f"line {call.lineno}" for call in _calls("setToolTip")
            if "Fisher's skewness" in src[call.lineno - 1]}

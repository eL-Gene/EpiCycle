"""I-11: ``QCPlotWidget`` in its classic layout is byte-identical to today.

F.  The Workbench's plot changes are opt-in
(``layout="workbench"``); the default ``layout="classic"`` is Step 3's QC
plot and must build the same widget tree, texts and styles as before the
revamp.  The golden ``golden/qc_plot_classic.json`` was generated from the
UNTOUCHED widget (brick B0's first commit) and must never be regenerated to
make a later brick pass: a difference here is a Step 3 regression.

Regenerate only on purpose, with ``EPICYCLE_REGEN_GOLDEN=1``.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

GOLDEN = Path(__file__).parent / "golden" / "qc_plot_classic.json"


_APP = None


def _app():
    """The QApplication, held at module level so it is never collected."""
    global _APP
    from qtpy.QtWidgets import QApplication
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _describe(widget) -> dict:
    """One widget's visible contract: class, name, texts, tooltip, style."""
    from qtpy.QtWidgets import QComboBox

    d = {
        "class": type(widget).__name__,
        "object_name": widget.objectName(),
        "hidden": bool(widget.isHidden()),
        "tooltip": widget.toolTip(),
        "style": widget.styleSheet(),
    }
    text = getattr(widget, "text", None)
    if callable(text):
        try:
            value = text()
            if isinstance(value, str):
                d["text"] = value
        except TypeError:
            pass
    if isinstance(widget, QComboBox):
        d["items"] = [widget.itemText(i) for i in range(widget.count())]
        d["current"] = widget.currentIndex()
    for attr in ("isCheckable", "isChecked"):
        fn = getattr(widget, attr, None)
        if callable(fn):
            try:
                d[attr] = bool(fn())
            except TypeError:
                pass
    return d


def _tree(widget) -> dict:
    from qtpy.QtWidgets import QWidget

    node = _describe(widget)
    node["children"] = [
        _tree(c) for c in widget.children() if isinstance(c, QWidget)
    ]
    return node


def _build_classic():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    params = inspect.signature(QCPlotWidget.__init__).parameters
    if "layout" in params:
        return QCPlotWidget(layout="classic")
    return QCPlotWidget()


def test_classic_layout_matches_the_golden():
    _app()
    widget = _build_classic()
    try:
        tree = _tree(widget)
    finally:
        widget.deleteLater()
    if os.environ.get("EPICYCLE_REGEN_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(tree, indent=1, ensure_ascii=False),
                          encoding="utf-8")
        pytest.skip("golden regenerated")
    assert GOLDEN.exists(), "golden missing: regenerate on UNTOUCHED code"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert tree == expected


def test_default_constructor_is_the_classic_layout():
    """No keyword keeps Step 3's plot: its call site passes nothing."""
    _app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    a = QCPlotWidget()
    b = _build_classic()
    try:
        assert _tree(a) == _tree(b)
    finally:
        a.deleteLater()
        b.deleteLater()

"""I-11 / D-f: ``QCPlotWidget()`` -- the classic
layout Step 3 builds -- keeps the same widget tree, texts, styles and
signals as before the Step 4 Workbench revamp.

The fixture ``golden/qc_plot_classic_layout.json`` was written from the
UNTOUCHED ``qc_plot_widget.py`` (the base commit of brick B3a), before any
``layout="workbench"`` code existed.  Every Workbench change sits behind
``layout="workbench"``; this test proves the default did not move.

Regenerate ONLY on a deliberate Step 3 change, with the owner's word::

    EPICYCLE_WRITE_GOLDEN=1 python -m pytest tests/fusion_reconstruction/test_qc_plot_classic_layout_golden.py -o addopts=""
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).with_name("golden") / "qc_plot_classic_layout.json"


_APP = []


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    _APP.append(app)  # a dropped QApplication is destroyed at once
    return app


def _df(n=24, seed=3):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "area": rng.rand(n) * 100.0 + 1.0,
        "solidity": rng.rand(n),
        "qc_score": rng.rand(n),
        "qc_tier": rng.choice(["good", "review", "bad"], size=n),
        "is_border_cell": rng.rand(n) < 0.2,
    })


def _widget_record(w):
    from qtpy.QtWidgets import QAbstractButton, QComboBox, QLabel

    rec = {
        "cls": type(w).__name__,
        "name": w.objectName(),
        "hidden": w.isHidden(),
        "tip": w.toolTip(),
        "style": w.styleSheet(),
        "minw": w.minimumWidth(),
        "maxw": w.maximumWidth(),
        "minh": w.minimumHeight(),
        "maxh": w.maximumHeight(),
    }
    if isinstance(w, (QLabel, QAbstractButton)):
        rec["text"] = w.text()
    if isinstance(w, QAbstractButton):
        rec["checkable"] = w.isCheckable()
        rec["checked"] = w.isChecked()
    if isinstance(w, QComboBox):
        rec["items"] = [w.itemText(i) for i in range(w.count())]
        rec["current"] = w.currentText()
        rec["editable"] = w.isEditable()
    return rec


def _layout_record(lay):
    out = {"cls": type(lay).__name__, "spacing": lay.spacing(),
           "margins": list(lay.getContentsMargins()), "items": []}
    for i in range(lay.count()):
        it = lay.itemAt(i)
        if it.widget() is not None:
            wd = it.widget()
            out["items"].append(["w", type(wd).__name__,
                                 getattr(wd, "text", lambda: "")()
                                 if hasattr(wd, "text") else ""])
        elif it.layout() is not None:
            out["items"].append(["l", _layout_record(it.layout())])
        else:
            out["items"].append(["s"])
    return out


def _plot_texts(pw):
    import pyqtgraph as pg

    return [it.toPlainText() for it in pw._plot_widget.getPlotItem().items
            if isinstance(it, pg.TextItem)]


def snapshot():
    """The classic plot's tree, texts, styles and signals, in several states."""
    _app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    pw = QCPlotWidget()
    snap = {}
    own = [c for c in pw.findChildren(type(pw).__mro__[1])  # QWidget
           if not _inside_plot(c, pw)]
    snap["widgets"] = [_widget_record(c) for c in own]
    snap["layout"] = _layout_record(pw.layout())
    snap["background"] = pw._plot_widget.backgroundBrush().color().name()
    snap["signals"] = sorted(
        k for k, v in vars(QCPlotWidget).items()
        if type(v).__name__ in ("pyqtSignal", "Signal"))
    pw.set_features(_df())
    states = {}
    for ptype in QCPlotWidget.PLOT_TYPES:
        pw.plot_type_combo.setCurrentText(ptype)
        states[ptype] = {
            "x_label": pw._x_label.text(),
            "hidden": {n: getattr(pw, n).isHidden() for n in (
                "_y_label", "_feature_y_combo", "_select_btn",
                "_clear_sel_btn", "_stats_label")},
            "stats": pw._stats_label.text(),
            "legend": pw._legend_label.text(),
            "plot_texts": _plot_texts(pw),
        }
    snap["states"] = states
    pw.plot_type_combo.setCurrentText("Scatter")
    pw._color_combo.setCurrentText("area")
    pw._redraw()
    snap["feature_colour"] = {
        "legend": pw._legend_label.text(),
        "hidden": {n: getattr(pw, n).isHidden() for n in (
            "_norm_label", "_norm_combo", "_cmap_label", "_cmap_combo")},
    }
    pw.highlight_labels([1, 2, 3])
    snap["selected"] = {"stats": pw._stats_label.text(),
                        "clear_hidden": pw._clear_sel_btn.isHidden()}
    pw.deleteLater()
    return snap


def _inside_plot(child, pw):
    """True for pyqtgraph's own internals under the PlotWidget."""
    p = child.parent()
    while p is not None and p is not pw:
        if p is pw._plot_widget:
            return True
        p = p.parent()
    return False


def test_classic_layout_is_byte_identical():
    if os.environ.get("EPICYCLE_WRITE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(snapshot(), indent=1, ensure_ascii=False),
                          encoding="utf-8")
    assert GOLDEN.exists(), f"golden fixture missing: {GOLDEN}"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = json.loads(json.dumps(snapshot()))
    for key in expected:
        assert got.get(key) == expected[key], f"classic layout changed in {key!r}"
    assert set(got) == set(expected)


def test_classic_is_the_default_keyword():
    import inspect

    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    sig = inspect.signature(QCPlotWidget.__init__)
    if "layout" in sig.parameters:
        p = sig.parameters["layout"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default == "classic"

"""Shared snapshot helpers for the QC plot goldens (Step 3 plot pass, P1).

Font-free on purpose: no widths or heights that come from font metrics,
so a golden captured on one machine holds on another.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_APP = []


def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    a = QApplication.instance() or QApplication([])
    _APP.append(a)  # a dropped QApplication is destroyed at once
    return a


def five_tier_df(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """A deterministic 60-row table: five tiers + unassigned, ghosts,
    clusters with noise, border cells."""
    rng = np.random.RandomState(seed)
    tiers = ["good", "bad", "garbage", "review", "curated", ""]
    status = np.array(["active"] * n, dtype=object)
    for lid in (5, 17, 33, 48):
        status[lid - 1] = "deleted"
    status[41 - 1] = "superseded"
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "area": np.round(rng.rand(n) * 3000.0 + 100.0, 3),
        "solidity": np.round(rng.rand(n), 4),
        "qc_score": np.round(rng.rand(n), 4),
        "qc_tier": [tiers[i % len(tiers)] for i in range(n)],
        "label_status": status,
        "cluster_id": np.array([(i % 5) - 1 for i in range(n)], dtype=np.int64),
        "is_border_cell": (np.arange(n) % 7) == 0,
    })


_URL = __import__("re").compile(r'url\("?([^)"]*)"?\)')


def portable(style: str) -> str:
    """A stylesheet with every ``url(...)`` cut to the file's basename: the
    tick image's path is this checkout's absolute path (``theme.py`` builds
    it from ``__file__``), so a golden that kept it was red in any other
    worktree."""
    return _URL.sub(lambda m: 'url("' + m.group(1).replace("\\", "/")
                    .rsplit("/", 1)[-1] + '")', style or "")


def describe(widget) -> dict:
    """One widget's visible contract (no font-derived sizes)."""
    from qtpy.QtWidgets import QComboBox

    d = {
        "class": type(widget).__name__,
        "object_name": widget.objectName(),
        "hidden": bool(widget.isHidden()),
        "tooltip": widget.toolTip(),
        "style": portable(widget.styleSheet()),
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
    for attr in ("isCheckable", "isChecked", "isEnabled"):
        fn = getattr(widget, attr, None)
        if callable(fn):
            try:
                d[attr] = bool(fn())
            except TypeError:
                pass
    return d


def tree(widget) -> dict:
    from qtpy.QtWidgets import QWidget

    node = describe(widget)
    node["children"] = [
        tree(c) for c in widget.children() if isinstance(c, QWidget)
    ]
    return node


def plot_texts(pw) -> list:
    import pyqtgraph as pg

    return [it.toPlainText() for it in pw._plot_widget.getPlotItem().items
            if isinstance(it, pg.TextItem)]


def _rgba(c) -> list:
    return [c.red(), c.green(), c.blue(), c.alpha()]


def scatter_items(pw) -> list:
    """Every scatter the plot draws: z, count, first point's look."""
    import pyqtgraph as pg

    out = []
    for it in pw._plot_widget.getPlotItem().items:
        if not isinstance(it, pg.ScatterPlotItem):
            continue
        pts = it.data
        rec = {"z": float(it.zValue()), "n": int(len(pts))}
        if len(pts):
            rec["size"] = float(pts["size"][0]) if pts["size"][0] >= 0 \
                else float(it.opts["size"])
            sym = pts["symbol"][0]
            rec["symbol"] = sym if isinstance(sym, str) else str(
                it.opts["symbol"])
            brush = pts["brush"][0] if pts["brush"][0] is not None \
                else it.opts["brush"]
            pen = pts["pen"][0] if pts["pen"][0] is not None \
                else it.opts["pen"]
            rec["brush"] = _rgba(pg.mkBrush(brush).color())
            p = pg.mkPen(pen)
            rec["pen"] = _rgba(p.color()) + [round(float(p.widthF()), 3),
                                             int(getattr(p.style(), "value", p.style()))]
            rec["brushes"] = [_rgba(pg.mkBrush(b).color())
                              if b is not None else None
                              for b in pts["brush"]]
        out.append(rec)
    return out


def colours(pw, df) -> list:
    return np.round(np.asarray(pw._get_point_colors_for(df),
                               dtype=np.float64), 4).tolist()

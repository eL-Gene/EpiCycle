"""Step 3 plot pass: the Workbench
plot that Step 4 builds -- ``QCPlotWidget(layout="workbench",
texts=WORKBENCH_PLOT_TEXTS)`` -- stays byte-identical.

**This golden is Step 4's contract.**  ``golden/qc_plot_workbench.json``
was written from the UNTOUCHED ``qc_plot_widget.py`` (``main`` at
``04d9131e``), BEFORE brick P1 added ``palette=`` / ``tier_scheme=``.
Every P1 change is read only under ``layout="workbench"`` with Step 4's
values as the defaults, so nothing recorded here may move.  It is never
regenerated to make a later brick pass; a difference is a Step 4
regression.  Regenerate only on a deliberate Step 4 change, with the
owner's word::

    EPICYCLE_WRITE_GOLDEN_WORKBENCH=1 python -m pytest tests/fusion_reconstruction/test_qc_plot_workbench_golden.py -o addopts=""

New INSTANCE signals may be added (the comparison of ``instance_signals``
is a superset check); none may vanish.  The class signal list is exact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import numpy as np

from tests.fusion_reconstruction import _qc_plot_snap as S

GOLDEN = Path(__file__).with_name("golden") / "qc_plot_workbench.json"


def _instance_signals(pw) -> list:
    return sorted(k for k, v in vars(pw).items()
                  if type(v).__name__ in ("pyqtBoundSignal",
                                          "SignalInstance"))


def snapshot() -> dict:
    S.app()
    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_PLOT_TEXTS,
    )

    QCPlotWidget = M.QCPlotWidget
    pw = QCPlotWidget(layout="workbench", texts=WORKBENCH_PLOT_TEXTS)
    pw.resize(562, 600)
    snap: dict = {}
    snap["tree_built"] = S.tree(pw)
    snap["styles"] = {name: getattr(M, name) for name in (
        "_WB_COMBO_STYLE", "_WB_LABEL_STYLE", "_WB_SEG_STYLE",
        "_WB_TOGGLE_STYLE", "_WB_CLEAR_STYLE", "_WB_BG")}
    snap["consts"] = {
        "WB_TEXT_KEYS": list(M.WB_TEXT_KEYS),
        "WB_HELP_KEYS": list(M.WB_HELP_KEYS),
        "WB_TIER_BUCKETS": list(M.WB_TIER_BUCKETS),
        "WB_PLOT_MIN_H": M.WB_PLOT_MIN_H,
        "WB_PLOT_MAX_H": M.WB_PLOT_MAX_H,
        "bucket_of": {t: M.wb_tier_bucket(t) for t in (
            "good", "bad", "garbage", "review", "curated", "", None)},
    }
    snap["class_signals"] = sorted(
        k for k, v in vars(QCPlotWidget).items()
        if type(v).__name__ in ("pyqtSignal", "Signal"))
    snap["instance_signals"] = _instance_signals(pw)
    snap["history_checked_built"] = pw._history_cb.isChecked()

    df = S.five_tier_df()
    stats_seen = []
    pw.stats_changed.connect(lambda: stats_seen.append(pw._stats_label.text()))
    pw.set_features(df)
    snap["history_checked_loaded"] = pw._history_cb.isChecked()
    snap["colour_items"] = [pw._color_combo.itemText(i)
                            for i in range(pw._color_combo.count())]

    per_type = {}
    for ptype in QCPlotWidget.PLOT_TYPES:
        pw.plot_type_combo.setCurrentText(ptype)
        per_type[ptype] = {
            "stats": pw._stats_label.text(),
            "legend": pw._legend_label.text(),
            "plot_texts": S.plot_texts(pw),
            "axes_note": pw._wb_axes_note.text(),
        }
    snap["per_type"] = per_type
    pw.plot_type_combo.setCurrentText("Scatter")

    colour_states = {}
    for name in ("QC tier", "Cluster", "QC status", "Border cell"):
        assert pw.set_colour_by(name), name
        ids, rgba = pw.viewer_colours()
        colour_states[name] = {
            "legend": pw._legend_label.text(),
            "legend_hidden": pw._legend_label.isHidden(),
            "gradient_hidden": pw._wb_gradient.isHidden(),
            "ids": ids.tolist(),
            "rgba": np.round(rgba.astype(np.float64), 4).tolist(),
            "scatter": S.scatter_items(pw),
        }
    for norm in ("Linear", "Log", "Rank"):
        pw._norm_combo.setCurrentText(norm)
        assert pw.set_colour_by("area")
        ids, rgba = pw.viewer_colours()
        g = pw._wb_gradient
        colour_states["area/" + norm] = {
            "legend": pw._legend_label.text(),
            "legend_hidden": pw._legend_label.isHidden(),
            "gradient_hidden": g.isHidden(),
            "gradient": [g.name_label.text(), g.lo_label.text(),
                         g.hi_label.text(), g.scale_label.text()],
            "ids": ids.tolist(),
            "rgba": np.round(rgba.astype(np.float64), 4).tolist(),
        }
    snap["colour_states"] = colour_states

    # Ghosts drawn as today when the toggle is ticked (Step 4: "Removed").
    pw.set_colour_by("QC tier")
    pw._history_cb.setChecked(True)
    snap["ghosts_on"] = {
        "legend": pw._legend_label.text(),
        "scatter": S.scatter_items(pw),
    }
    pw._history_cb.setChecked(False)

    # A legend pick on a bucket, then a plot-type switch that keeps it.
    pw._toggle_category_selection("qc_tier", "bad")
    pick = {"stats": pw._stats_label.text(),
            "selected": pw.selected_label_ids(),
            "legend": pw._legend_label.text(),
            "clear_enabled": pw._wb_clear_btn.isEnabled()}
    pw.plot_type_combo.setCurrentText("PCA")
    pick["after_switch_selected"] = pw.selected_label_ids()
    pick["after_switch_categories"] = sorted(
        "/".join(map(str, k)) for k in pw._selected_categories)
    pw.plot_type_combo.setCurrentText("Scatter")
    pw._toggle_category_selection("qc_tier", "unmarked")
    pick["unmarked_selected"] = pw.selected_label_ids()
    snap["pick"] = pick
    snap["stats_seen_count"] = len(stats_seen)
    snap["tree_loaded"] = S.tree(pw)
    pw.deleteLater()
    return snap


def test_workbench_plot_is_byte_identical():
    if os.environ.get("EPICYCLE_WRITE_GOLDEN_WORKBENCH") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(snapshot(), indent=1,
                                     ensure_ascii=False), encoding="utf-8")
        pytest.fail(f"golden written: {GOLDEN.name} (unset EPICYCLE_WRITE_GOLDEN_WORKBENCH)")
    assert GOLDEN.exists(), f"golden fixture missing: {GOLDEN}"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = json.loads(json.dumps(snapshot()))
    assert set(got) == set(expected)
    for key in expected:
        if key == "instance_signals":
            assert set(expected[key]) <= set(got[key]), key
            continue
        assert got[key] == expected[key], f"Workbench plot changed in {key!r}"

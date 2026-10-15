"""Step 3 plot pass: the CLASSIC
plot's tier legend and point colours under Colour by "QC tier".

The two older classic goldens open on "QC status" and never record tier
colours; this one does.  **It is Step 3's classic contract**:
``golden/qc_plot_classic_tier_legend.json`` was written from the
UNTOUCHED ``qc_plot_widget.py`` (``main`` at ``04d9131e``), before brick
P1 added ``palette=`` / ``tier_scheme=``.  Those keywords are read only
under ``layout="workbench"``, so the classic layout must not move -- not
even when they are passed (``test_qc_plot_tier_scheme.py`` re-runs
:func:`snapshot` with them).  Never regenerated to make a brick pass.
Regenerate only on a deliberate Step 3 change, with the owner's word (one variable per golden, and a
write run fails on purpose so it can never pass as a check)::

    EPICYCLE_WRITE_GOLDEN_CLASSIC_TIER_LEGEND=1 python -m pytest tests/fusion_reconstruction/test_qc_plot_classic_tier_legend_golden.py -o addopts=""
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.fusion_reconstruction import _qc_plot_snap as S

GOLDEN = Path(__file__).with_name("golden") / "qc_plot_classic_tier_legend.json"


def snapshot(**kwargs) -> dict:
    """The classic plot under QC tier.  ``kwargs`` go to the constructor."""
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    pw = QCPlotWidget(**kwargs)
    pw.resize(562, 600)
    df = S.five_tier_df()
    snap = {"history_built": pw._history_cb.isChecked()}
    pw.set_features(df)
    snap["colour_items"] = [pw._color_combo.itemText(i)
                            for i in range(pw._color_combo.count())]
    pw._color_combo.setCurrentText("QC tier")
    pw._redraw()
    snap["tier"] = {
        "legend": pw._legend_label.text(),
        "stats": pw._stats_label.text(),
        "history": pw._history_cb.isChecked(),
        "colours": S.colours(pw, df),
        "scatter": S.scatter_items(pw),
    }
    pw._history_cb.setChecked(False)
    snap["tier_no_ghosts"] = {
        "legend": pw._legend_label.text(),
        "scatter": S.scatter_items(pw),
    }
    pw._history_cb.setChecked(True)
    pw._toggle_category_selection("qc_tier", "review")
    snap["pick"] = {"selected": pw.selected_label_ids(),
                    "stats": pw._stats_label.text(),
                    "legend": pw._legend_label.text()}
    pw.plot_type_combo.setCurrentText("PCA")
    snap["after_switch"] = {"selected": pw.selected_label_ids(),
                            "legend": pw._legend_label.text()}
    snap["tree"] = S.tree(pw)
    pw.deleteLater()
    return snap


def test_classic_tier_legend_is_byte_identical():
    if os.environ.get("EPICYCLE_WRITE_GOLDEN_CLASSIC_TIER_LEGEND") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(snapshot(), indent=1,
                                     ensure_ascii=False), encoding="utf-8")
        pytest.fail(f"golden written: {GOLDEN.name} (unset EPICYCLE_WRITE_GOLDEN_CLASSIC_TIER_LEGEND)")
    assert GOLDEN.exists(), f"golden fixture missing: {GOLDEN}"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = json.loads(json.dumps(snapshot()))
    assert set(got) == set(expected)
    for key in expected:
        assert got[key] == expected[key], f"classic tier legend changed in {key!r}"

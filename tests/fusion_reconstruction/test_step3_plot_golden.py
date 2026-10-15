"""Step 3 plot pass: the right
column of Curation -- ``QCPlotArea`` -- as a golden: its widget tree,
styles, texts and checked / enabled states with no patch, with a patch
loaded (the load default: PCA, qc_score · Coolwarm · Rank) and with a
lasso selection, plus the area's own constants.

**This golden is Step 3's contract from brick P2 on.**  It was captured at
the END of P2, after the live driver.  A later brick that changes what the
area shows regenerates it on purpose, with the owner's word, in the same
commit as the change::

    EPICYCLE_WRITE_GOLDEN_STEP3_PLOT=1 python -m pytest tests/fusion_reconstruction/test_step3_plot_golden.py -o addopts=""

Font-free (``_qc_plot_snap.describe`` records no font-derived sizes).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.fusion_reconstruction._platform import windows_font_metrics

from tests.fusion_reconstruction import _qc_plot_snap as S

GOLDEN = Path(__file__).with_name("golden") / "step3_plot_area.json"


def _pump():
    from qtpy.QtWidgets import QApplication

    for _ in range(3):
        QApplication.processEvents()


def snapshot() -> dict:
    S.app()
    from napari_epicycle.fusion_reconstruction import _qc_plot_area as A

    area = A.QCPlotArea()
    area.resize(640, 760)
    snap: dict = {}
    snap["consts"] = {
        name: getattr(A, name) for name in (
            "PLOT_DEFAULT_PX", "LOWER_SLACK_PX", "HANDLE_PX", "TIERS",
            "UNASSIGNED", "GHOST_STATUSES", "STATUS_INFO_COLOUR",
            "STATUS_WARN_COLOUR", "STATUS_ERROR_COLOUR", "STATUS_MAX_SHARE",
            "SWITCH_TO_SELECTION", "NO_SWITCH", "SOURCES")}
    snap["consts"] = {k: list(v) if isinstance(v, tuple) else v
                      for k, v in snap["consts"].items()}
    snap["styles"] = {name: S.portable(getattr(A, name)) for name in dir(A)
                      if name.startswith("_") and name.endswith("_STYLE")}
    snap["styles"]["card_on"] = A._card_style(A._CARD_EDGE_ON)
    snap["styles"]["card_none"] = A._card_style(A._CARD_EDGE_NONE)
    snap["tree_empty"] = S.tree(area)

    df = S.five_tier_df()
    pw = area.qc_plot_widget
    pw.set_features(df)
    pw.plot_type_combo.setCurrentText("PCA")
    pw.set_colour_by("qc_score", colours="Coolwarm", scale="Rank")
    area.panel.set_data(df)
    area.set_dataframe(df)
    area.set_enabled_state(True, False, False)
    _pump()
    snap["tree_loaded"] = S.tree(area)

    ids = [int(x) for x in df.loc[df.label_status == "active", "label"]][:9]
    pw.highlight_labels(ids)
    area.selection_changed("lasso")
    area.set_enabled_state(True, True, False)
    _pump()
    snap["tree_selection"] = S.tree(area)
    snap["card_selection"] = {
        "count": area.card.count_label.text(),
        "chips": area.card.chips_label.text(),
        "message": list(area.card.message()),
        "totals": area.card.totals_label.text(),
        "mix": [list(x) for x in area.card.mix_bar.shares()],
    }
    area.deleteLater()
    _pump()
    return json.loads(json.dumps(snap, default=str))


@windows_font_metrics
def test_the_step3_plot_area_matches_its_golden():
    snap = snapshot()
    if os.environ.get("EPICYCLE_WRITE_GOLDEN_STEP3_PLOT"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True),
                          encoding="utf-8")
        pytest.fail("golden written -- rerun without "
                    "EPICYCLE_WRITE_GOLDEN_STEP3_PLOT to check it")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for key in sorted(set(want) | set(snap)):
        assert snap.get(key) == want.get(key), key


def test_the_golden_is_checkout_independent():
    # P3 found it red in a sibling worktree: the tick image's url() in
    # PANEL_CHECKBOX_STYLE carried this checkout's absolute path.
    text = GOLDEN.read_text(encoding="utf-8")
    assert "C:/Users" not in text and "EpiCycle-plotpass" not in text
    assert 'url(\\"_panel_tick.svg\\")' in text

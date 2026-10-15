"""Height budget (brick P2): the right column of
Curation -- the "QC Plot" header, the plot area, the busy bar and the
status line, built as ``ui.py`` builds it -- measured at the column's REAL
widths (P2 audit HIGH-2, measured live): 562 px with a 1000 px dock (the
owner's working width) and 447 px with a 640 px dock, idle and busy, with
the Histogram view and with the Selection view.  Since the audit the area
ASKS for the plot's default 270 px (not its 150 px divider floor) and the
lower area's floor + 30 px, so these are the heights the Curation section
scrolls at.

Measured in a SUBPROCESS with the real font (``QT_QPA_FONTDIR`` and
Segoe UI 9): offscreen Qt draws every glyph about 1.6x too
wide otherwise, and changing the font in-process leaks into later tests.

The values asserted are the ones measured when P2 was built.  They are
CEILINGS: a change that makes the column taller fails here, and a ceiling
is never raised without the owner.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: (width, state, view) -> minimumSizeHint().height() of the column, px.
CEILINGS = {
    # measured after the P2 audits (2026-10-06, Segoe UI 9): the plot's
    # default 270 px + the lower area's floor 210 px (tabs 26 + 6 + the
    # histogram's 158 + 20 for its note) + 30 px slack + the chrome at that
    # width.  The divider's own floors stay 150 / 210.
    "562/idle/histogram": 794,
    "562/idle/selection": 794,
    "562/busy/histogram": 794,
    "562/busy/selection": 794,
    "447/idle/histogram": 858,
    "447/idle/selection": 858,
    "447/busy/histogram": 858,
    "447/busy/selection": 858,
}

_SCRIPT = r'''
import json, os, sys
from qtpy.QtWidgets import (QApplication, QLabel, QProgressBar, QVBoxLayout,
                            QWidget)
from qtpy.QtGui import QFont
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
sys.path.insert(0, os.environ["EPI_REPO"])
from tests.fusion_reconstruction import _qc_plot_snap as S
from napari_epicycle.fusion_reconstruction.ui import _make_subsection_header
from napari_epicycle.fusion_reconstruction._qc_plot_area import QCPlotArea
from napari_epicycle.ui_shell.theme import (PROGRESS_BAR_STYLE,
                                            STATUS_LABEL_STYLE)

def pump():
    for _ in range(6):
        app.processEvents()


def min_height_for_width(lay, width):
    """The layout's honest minimum height at ``width``: a word-wrapped
    label counts its wrapped lines (Qt's minimumSizeHint does not)."""
    m = lay.contentsMargins()
    inner = max(1, width - m.left() - m.right())
    total = m.top() + m.bottom()
    n = 0
    for i in range(lay.count()):
        item = lay.itemAt(i)
        w = item.widget()
        if w is not None and w.isHidden():
            continue
        if item.isEmpty() and item.layout() is None:
            continue
        n += 1
        if item.layout() is not None:
            total += max(min_height_for_width(item.layout(), inner),
                         item.minimumSize().height())
        elif item.hasHeightForWidth():
            total += item.heightForWidth(inner)
        elif w is not None:
            total += max(w.minimumSizeHint().height(), w.minimumHeight())
        else:
            total += item.minimumSize().height()
    return int(total + max(0, lay.spacing()) * max(0, n - 1))

out = {}
for width in (562, 447):
    for state in ("idle", "busy"):
        for view in ("histogram", "selection"):
            col = QWidget()
            lay = QVBoxLayout(col)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(4)
            lay.addLayout(_make_subsection_header(col, "QC Plot",
                          guide_filename="step3_curation_qc_plot.md"))
            area = QCPlotArea(col)
            lay.addWidget(area, 1)
            bar = QProgressBar(col)
            bar.setStyleSheet(PROGRESS_BAR_STYLE)
            bar.setTextVisible(True)
            lay.addWidget(bar)
            status = QLabel(col)
            status.setStyleSheet(STATUS_LABEL_STYLE)
            status.setWordWrap(True)
            lay.addWidget(status)
            df = S.five_tier_df()
            pw = area.qc_plot_widget
            pw.set_features(df)
            area.panel.set_data(df)
            area.set_dataframe(df)
            # the load default: qc_score, Coolwarm, Rank (a feature colour
            # shows Scale + Colours and the gradient legend: the tallest
            # chrome the load can show)
            pw.set_colour_by("qc_score", colours="Coolwarm", scale="Rank")
            if view == "selection":
                ids = [int(x) for x in df.loc[df.label_status == "active",
                                              "label"]][:9]
                pw.highlight_labels(ids)
                area.selection_changed("lasso")
            if state == "busy":
                bar.setRange(0, 0)
                status.setText("Loading Wound_patch_1 — reading the "
                               "volume and downsampling it for the view…")
                area.set_locked(True)
            else:
                bar.setRange(0, 100)
                status.setText("Curating: Wound_patch_1")
            col.resize(width, 1400)
            col.show()
            pump()
            area._update_pinned_minimum()
            pump()
            key = f"{width}/{state}/{view}"
            out[key] = {
                "column": min_height_for_width(lay, width),
                "column_qt": col.minimumSizeHint().height(),
                "column_w": col.minimumSizeHint().width(),
                "area": area.minimumSizeHint().height(),
                "pinned_min": area._pinned.minimumHeight(),
                "lower_min": area._lower.minimumHeight(),
                "card": area.card.minimumSizeHint().height(),
            }
            col.hide()
            col.deleteLater()
            pump()
print("CEILING-JSON " + json.dumps(out))
'''


def _measure() -> dict:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_QPA_FONTDIR"] = "C:/Windows/Fonts"
    env["EPI_REPO"] = str(REPO)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], env=env,
                          capture_output=True, text=True, timeout=600,
                          cwd=str(REPO))
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("CEILING-JSON ")), None)
    assert line, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(line[len("CEILING-JSON "):])


@pytest.fixture(scope="module")
def measured():
    if not Path("C:/Windows/Fonts/segoeui.ttf").exists():
        pytest.skip("needs the real Segoe UI font")
    m = _measure()
    for k, v in sorted(m.items()):
        print(f"[QC-AREA-CEILING] {k}: {v}")
    return m


@pytest.mark.parametrize("key", sorted(CEILINGS))
def test_the_column_is_within_its_ceiling(measured, key):
    got = measured[key]["column"]
    assert got <= CEILINGS[key], (key, got, CEILINGS[key])


@pytest.mark.parametrize("key", sorted(CEILINGS))
def test_nothing_widens_the_column(measured, key):
    width = int(key.split("/")[0])
    assert measured[key]["column_w"] <= width, measured[key]


def test_the_selection_card_fits_the_lower_floor(measured):
    for key, v in measured.items():
        assert v["card"] <= v["lower_min"], (key, v)

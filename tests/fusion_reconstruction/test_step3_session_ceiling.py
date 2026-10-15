"""Session's height, measured where the user gets it (A2 "the Session
ceiling re-measured"; board B).

Session had no ceiling of its own before A2; A2 rebuilds it (the two cards,
the header count line, "Use another folder…" under the list, the rows' eye,
inline "Set spacing…" and tier counts, the Pipeline status block under the
list) and pins it here.  The section's viewport is 566 px (the 578 px
default dock) and 1000 px; the panel is measured at that width in two
states: A (an empty workspace) and B (ten patches -- curated rows with
their tier counts, an amber row with its button, the header line with its
amber clause; the list at its cap, scrolling inside itself).  The
numbers come from a SUBPROCESS with the real font.

Measured 2026-10-10 (Segoe UI 9 pt), the A2 re-baseline: A 428 / B 656 px
at 566, A 415 / B 643 px at 1000; the list 300 px holding 591 px of rows
in B; the panel's minimum width 276 px.
A3 re-baseline (2026-10-10, the rows v2): B 656 -> 906 px at 566 and
643 -> 893 px at 1000 -- the list's cap is four rows of the 128 px picture
(550 px, was 300), the rows hold 1,360 px; A is unchanged.  The panel's
minimum width in B is 404 px (the picture, the text and the eye).
A later brick that grows Session fails here and must say why.  Run one file
at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.fusion_reconstruction._platform import windows_font_metrics

_FONT_DIR = "C:/Windows/Fonts"
_REPO = Path(__file__).resolve().parents[2]
VIEWPORTS = (566, 1000)

#: measured (A2 re-baseline): {viewport: {state: panel height in px}}
CEILINGS = {
    566: {"A": 428, "B": 906},
    1000: {"A": 415, "B": 893},
}
#: the panel's own minimum width (it must fit the 566 px viewport)
MIN_W = 276

TIERS = (("good", 412, (80, 200, 120)), ("review", 96, (255, 200, 60)),
         ("bad", 31, (255, 120, 40)), ("garbage", 12, (240, 60, 60)))


def _hfw(widget, width):
    if widget.hasHeightForWidth():
        return widget.heightForWidth(width)
    return widget.sizeHint().height()


def measure_all(app) -> dict:
    """Runs in the subprocess."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QListWidgetItem, QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        SESSION_LIST_MAX_H,
        FusionReconstructionCuration,
    )

    def pump(n=5):
        for _ in range(n):
            app.processEvents()

    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    w.resize(700, 2400)
    w.show()
    w.switch_to_section(0)
    pump(10)
    sec = w._accordion._sections[0]
    panel = w._session_panel
    out = {"font": app.font().family(),
           "min_w": panel.minimumSizeHint().width(),
           "list_max": SESSION_LIST_MAX_H}

    def state_b():
        names = [f"WholeScale_patch_{i}.tiff" for i in range(1, 11)]
        statuses = {n: s for n, s in zip(names, (
            "curating", "curating", "unprocessed", "curating",
            "reconstructed", "reconstructed", "predicted", "reconstructed",
            "finalized", "accepted"))}
        lst = w.loaded_patch_list
        lst.clear()
        for n in names:
            it = QListWidgetItem(n)
            it.setData(Qt.UserRole, n)
            lst.addItem(it)
        sp = {n: {"zyx": [1.0, 0.33, 0.33], "source": "step1"}
              for n in names if statuses[n] != "unprocessed"}
        w.update_patch_list_status(
            statuses, axes_by_patch={n: "CZYX" for n in names},
            shapes={n: [2, 147, 240, 240] for n in names},
            info_by_patch={n: {"labels_shape": [147, 240, 240],
                               "num_cells_final": 3988}
                           for n in names if statuses[n] != "unprocessed"
                           and statuses[n] != "predicted"},
            spacing_by_patch=sp)
        w.set_tier_counts({n: TIERS for n in names
                           if statuses[n] in ("curating", "finalized",
                                              "accepted")}, replace=True)
        w.set_session_header_counts(10, 3, 1, 1)
        w.set_patch_dir_source(True, "")
        # A2 fold M4: a 60-character patch name on the Resume link
        w.refresh_session_activity(
            "3 days ago", "WholeScale_patch_with_a_very_long_name_0123456"
            "789abcde.tiff  (curating)", True)
        pump(10)

    for vp in VIEWPORTS:
        for _ in range(4):           # settle the dock so the viewport is vp
            delta = vp - sec._scroll.viewport().width()
            if delta == 0:
                break
            w.resize(w.width() + delta, 2400)
            pump(5)
        got = {"viewport": sec._scroll.viewport().width()}
        w.loaded_patch_list.clear()
        w.set_session_header_counts(0)
        w._fit_session_list_height()
        pump(5)
        got["A"] = _hfw(panel, vp)
        state_b()
        got["B"] = _hfw(panel, vp)
        got["B_min_w"] = panel.minimumSizeHint().width()
        got["B_list_h"] = w.loaded_patch_list.height()
        got["B_rows_h"] = sum(w.loaded_patch_list.sizeHintForRow(i)
                              for i in range(w.loaded_patch_list.count()))
        out[str(vp)] = got
    return out


_SUBPROCESS = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
import napari_epicycle  # the import order the app uses
from tests.fusion_reconstruction import test_step3_session_ceiling as t
print("S3JSON" + json.dumps(t.measure_all(app)), flush=True)
os._exit(0)     # the numbers are out; skip a Qt teardown that can crash
"""


@pytest.fixture(scope="module")
def measured():
    code = f"FONT_DIR = {_FONT_DIR!r}\n" + _SUBPROCESS
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("S3JSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:],
                   proc.stderr[-3000:])
    got = json.loads(lines[-1][len("S3JSON"):])
    print(f"[CEIL] {json.dumps(got)}")
    return got


def test_measured_in_the_users_font(measured):
    if sys.platform == "win32" and os.path.isdir(_FONT_DIR):
        assert measured["font"] == "Segoe UI"


@pytest.mark.parametrize("vp", VIEWPORTS)
def test_the_viewport_is_the_one_asked_for(measured, vp):
    assert measured[str(vp)]["viewport"] == vp


@pytest.mark.parametrize("vp", VIEWPORTS)
@pytest.mark.parametrize("state", ["A", "B"])
def test_session_is_no_taller_than_measured(measured, vp, state):
    h = measured[str(vp)][state]
    print(f"[CEIL] session {vp} {state}: {h} px")
    assert h <= CEILINGS[vp][state]


@pytest.mark.parametrize("vp", VIEWPORTS)
def test_the_list_stops_at_its_cap_and_scrolls(measured, vp):
    got = measured[str(vp)]
    assert got["B_list_h"] == measured["list_max"]
    assert got["B_rows_h"] > got["B_list_h"]       # it scrolls inside


@windows_font_metrics
def test_the_panel_fits_the_default_dock(measured):
    assert measured["min_w"] <= MIN_W <= 566


@pytest.mark.parametrize("vp", VIEWPORTS)
def test_a_long_resume_name_never_widens_the_panel(measured, vp):
    """A2 fold M4: state B carries a 60-character name on Resume."""
    assert measured[str(vp)]["B_min_w"] <= 566

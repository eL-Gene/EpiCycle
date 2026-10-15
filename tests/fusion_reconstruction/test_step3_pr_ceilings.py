"""The settings column's height, measured where the user gets it
(brick R2a).

R2a pins the COLUMN only (R3 owns the whole section's ceilings, once the
column moves into the right half of Step 4's 224 | 330 splitter).  The
panel is laid out so the Predict + Reconstruct section's viewport is 566 px
(the 578 px default dock) and 1000 px wide; the column is measured at the
width it then gets, in four states: D (Advanced closed) and E (Advanced
open with ONE group open), each idle and with every optional note showing.
E is measured with EACH of the seven u-Segment3D groups opened in turn and
the MAX is asserted (audit A MED-2: only "Smooth before joining" was
measured before); the per-group heights are reported too.  The notes: the
note slot under Start from with its LONGEST sentence (the last-run note or
the refused-pick sentence); the flow note under "Smooth before joining" and
Size limits' two dim notes ("Size limits are off..." and the group's
description) are on by default and stay on while measured.  The numbers come from a SUBPROCESS with the real
font (offscreen Qt draws every glyph ~1.6x too wide without
a font directory, and changing the app font leaks between tests).

The values below are the MEASURED ones (2026-10-06, Segoe UI 9 pt); a
later brick that grows the column fails here and must say why.  Re-measured
at the R2a fold with every group: the tallest E is "Follow the flows"
(``gradient_descent``) -- 1027 px at 566 (Smooth before joining 939,
Combine the probabilities 994, Size limits with its two notes 951, Clean
the mask 873, Combine the flows 840, Group into cells 805) and 965 px at
1000; at R3's 330 px 991.  The note slot's longest sentence (the
refused-pick one) wraps to two lines at 239 px: D_notes 358, E_notes 1060.  Since
R3 the column lives in the right half of the splitter (316 px wide at the 566
dock, 750 at 1000), so the 566 / 1000 numbers above are upper bounds kept as
R2a's pins; the SECTION's ceilings, R3's, are in
``test_step3_pr_layout_ceilings.py``.  Since the cell width's two lines sit
under the width row (R2b part 2, placed by R3's ``_build_pr_settings``) the
notes states show them too -- the "≈ N pixels" line and the no-voxel
warning: +47 px at 316, 330 and 750 px wide (D_notes 348 -> 395 and E_notes
1024 -> 1071 at 330 px; at 1000 D_notes 322 -> 369, E_notes 985 -> 1032);
idle is unchanged (both lines hidden).  Run one file at a time with
``-o addopts=""``.
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

#: measured: {viewport: {state: height in px}}, and the column's own
#: minimum width (it must fit R3's 330 px right column)
CEILINGS = {
    # A3 (2026-10-10, the 240 | 330 columns): the settings' scroll viewport
    # is 300 px where it was 316, so E_notes at 566 wraps further, 1071 ->
    # 1087 (D, D_notes and E measured the same)
    #
    # Brick B (2026-10-10, Predict + Reconstruct parity): the dim note under
    # the two ticks, "Outline tolerance and cell confidence are not used by
    # the 3D build, so they are not shown." (two lines at 300 px), re-pinned
    # to the measured values: 566 D 325 -> 348, D_notes 395 -> 428, E 1027 ->
    # 1040, E_notes 1087 -> 1120; 1000 D 302 -> 322, D_notes 369 -> 389, E
    # 965 -> 985, E_notes 1032 -> 1052; at R3's 330 px D 315 -> 348, D_notes
    # 395 -> 428, E 991 -> 1024, E_notes 1071 -> 1104.
    566: {"D": 348, "D_notes": 428, "E": 1040, "E_notes": 1120},
    1000: {"D": 322, "D_notes": 389, "E": 985, "E_notes": 1052},
}
#: at R3's right-column width (the column will live there from R3)
AT_R3 = {"D": 348, "D_notes": 428, "E": 1024, "E_notes": 1104}
MIN_W = 239
R3_RIGHT_COL_W = 330


def _hfw(widget, width):
    if widget.hasHeightForWidth():
        return widget.heightForWidth(width)
    return widget.sizeHint().height()


def measure_all(app) -> dict:
    """Runs in the subprocess."""
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
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
    pump(10)
    sec = w._accordion._sections[1]
    col = w._pr_settings
    out = {"font": app.font().family(),
           "min_w": col.minimumSizeHint().width()}
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P

    # the longest sentence the note slot under Start from can carry
    note_texts = (P.S3_LAST_RUN_NOTE, P.S3_LAST_RUN_VANISHED,
                  getattr(P, "S3_USEG3D_UNAVAILABLE", ""))
    w.pr_last_run_note.setWordWrap(True)
    # Size limits' two dim notes and the flow note show by default
    assert w.skip_postprocess_cb.isChecked()
    assert not w.ps_smooth_flows_cb.isChecked()
    groups = list(w._pr_adv_groups)

    # the cell width's two lines (R2b part 2): both showing in the notes
    # states -- the "≈ N pixels" line with a four-digit number and the
    # no-voxel warning
    cw_lines = w.pr_cell_width_lines
    cw_texts = (P.S3_CELL_WIDTH_PX.format(px=1000, um="0.10"),
                P.S3_NO_VOXEL_WARNING)

    def note_h(notes):
        """The column's height with the slot showing its tallest text."""
        if not notes:
            w.show_pr_last_run_note(False)
            cw_lines.show_lines("", "")
            pump(5)
            return _hfw(col, col.width()), _hfw(col, R3_RIGHT_COL_W)
        cw_lines.show_lines(*cw_texts)
        best = (0, 0)
        for text in note_texts:
            if not text:
                continue
            w.show_pr_last_run_note(True, text)
            pump(5)
            best = (max(best[0], _hfw(col, col.width())),
                    max(best[1], _hfw(col, R3_RIGHT_COL_W)))
        w.show_pr_last_run_note(False)
        cw_lines.show_lines("", "")
        return best

    for vp in VIEWPORTS:
        for _ in range(4):           # settle the dock so the viewport is vp
            delta = vp - sec._scroll.viewport().width()
            if delta == 0:
                break
            w.resize(w.width() + delta, 2400)
            pump(5)
        got = {"viewport": sec._scroll.viewport().width(), "groups": {}}
        at_r3 = out.setdefault("at_r3", {})
        for notes in (False, True):
            key = "D" + ("_notes" if notes else "")
            got[key], r3 = note_h(notes)
            at_r3[key] = max(at_r3.get(key, 0), r3)
        w._pr_adv_collapsible.expand(animate=False)
        for engine in groups:
            w._pr_adv_groups[engine].expand(animate=False)
            pump(5)
            for notes in (False, True):
                key = "E" + ("_notes" if notes else "")
                h, r3 = note_h(notes)
                got["groups"].setdefault(engine, {})[key] = h
                got[key] = max(got.get(key, 0), h)
                at_r3[key] = max(at_r3.get(key, 0), r3)
            w._pr_adv_groups[engine].collapse(animate=False)
            pump(5)
        w._pr_adv_collapsible.collapse(animate=False)
        pump(5)
        got["col_w"] = col.width()
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
from tests.fusion_reconstruction import test_step3_pr_ceilings as t
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
@pytest.mark.parametrize("state", ["D", "D_notes", "E", "E_notes"])
def test_the_column_is_no_taller_than_measured(measured, vp, state):
    h = measured[str(vp)][state]
    print(f"[CEIL] {vp} {state}: {h} px at column width "
          f"{measured[str(vp)]['col_w']}")
    assert h <= CEILINGS[vp][state]


@pytest.mark.parametrize("state", ["D", "D_notes", "E", "E_notes"])
def test_the_column_at_r3s_width_is_no_taller_than_measured(measured, state):
    h = measured["at_r3"][state]
    print(f"[CEIL] at {R3_RIGHT_COL_W} px {state}: {h} px")
    assert h <= AT_R3[state]


def test_every_group_was_measured(measured):
    """E is the max over the seven groups, not one group (audit A MED-2)."""
    for vp in VIEWPORTS:
        groups = measured[str(vp)]["groups"]
        assert len(groups) == 7
        for engine, hs in groups.items():
            print(f"[CEIL] {vp} E with {engine}: {hs}")
        assert measured[str(vp)]["E"] == max(h["E"] for h in groups.values())


def test_the_notes_cost_what_they_show(measured):
    for vp in VIEWPORTS:
        got = measured[str(vp)]
        assert got["D_notes"] > got["D"]
        assert got["E"] > got["D"]


@windows_font_metrics
def test_the_column_fits_r3s_right_column(measured):
    """Nothing in the column forces it wider than R3's 330 px column."""
    assert measured["min_w"] <= MIN_W
    assert MIN_W <= R3_RIGHT_COL_W

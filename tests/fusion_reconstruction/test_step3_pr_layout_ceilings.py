"""The Predict + Reconstruct SECTION's ceilings, measured where the user gets
them (brick R3).

R3 puts the section on Step 4's ② layout: the run row over a splitter
224 | 330 (minimums 200 | 300), the Model and Patches cards on the left, R2a's
settings column in a scroll area over the pinned foot on the right.  The
Step 3 widget is laid out so the section's viewport is 566 px (the 578 px
default dock) and then 1000 px wide, and measured in two boards -- D
(Advanced closed) and E (Advanced open with its tallest group, "Follow the
flows", open) -- each IDLE and with every optional line showing ("notes":
the foot's status as a two-line amber warning with the longest sentence a
run writes, the longest sentence of the note slot under Start from, the
model card's two lines with long names, ten patch rows, the cell width's
two lines -- the "≈ N pixels" line and the no-voxel warning).  Each ceiling is
the MAX over idle and notes, and both states are asserted against it.

What is measured, at each viewport and board:

* ``needed``: the whole section's height at the columns' REAL widths -- its
  chrome (margins, run row, separator: 55 px) plus the taller column.  Qt's
  own height-for-width of the section (``section``) asks the splitter's
  sizeHint, which lays the columns out at their PREFERRED widths; it is
  held to the same ceiling;
* ``right``: the settings column's content (it SCROLLS inside the right
  column), at the scroll viewport's width;
* ``foot``: the pinned foot;
* ``left_fixed``: the left column above and below the patch list;
* widths: nothing scrolls sideways -- the section's minimum width fits the
  viewport, the right content's fits its scroll viewport, the left column's
  fits 224 px; the left column IS 224 px and the right at least 300.

The numbers come from a SUBPROCESS with the real font (offscreen Qt draws every glyph ~1.6x too wide without a font directory, and
changing the app font leaks between tests).

MEASURED (2026-10-06, Segoe UI 9 pt, brick R3) -- these become the owner's
ceilings in the spec; a later brick that grows the section fails here and
must say why (idle / notes):

    viewport 566 (left 224, right column 324, its scroll viewport 316):
        D   needed 495 / 495   right 321 / 401    foot 111 / 142   left_fixed 248
        E   needed 495 / 495   right 997 / 1077   foot 111 / 142   left_fixed 248
    viewport 1000 (left 224, right column 758, its scroll viewport 750):
        D   needed 495 / 495   right 308 / 375    foot 111 / 124   left_fixed 248
        E   needed 495 / 495   right 971 / 1038   foot 111 / 124   left_fixed 248

V1 (2026-10-07), second pass: the board F state (the longest lock line
over the run row, the surface locked; the ticked lines give way to "You can
still click rows ...") is measured too, and the notes states paint ten rows
with a missing one and a ticked curating one.  ``needed`` 495 -> 423 at
both viewports for D and E: the patch list asks for 120 px instead of Qt's
192 (it still has the column's stretch) and the settings' scroll area asks
for nothing (it scrolls inside its column) -- on the live 848 px dock the
section scrolled under a two-line lock line before (measured: 547 px of
content in a 500 px viewport).  F measures needed 447 (566) / 432
(1000) and left_fixed 238 (the list note replaces the two ticked lines and
the link).

V1 (2026-10-07, the owner's mockup parity): ``left_fixed`` 248 -> 260 at
both viewports and in every state -- the left column takes the mockup's
gaps (the column's 6 px between the cards, the separator's own 4 px margin
above and below it, the cards' 4 px between their lines) where R3 had 4 px
everywhere; the model lines lost their 2 px padding and sit as one block.
``needed`` is unchanged (the patch list gives up the height).

The LEFT column decides the section's height (440 px: 248 of cards and lines
plus the patch list at its 192 px size hint; the right column is 393-419);
the section does not grow with Advanced -- the settings scroll inside the
right column, which is the point of the layout.  Minimum widths: the
section 518 at both viewports, the settings content 239 (in a 316 / 750 px
scroll viewport), the left column 130 (in 224).  The cell width's two lines
(R2b part 2, row 10) sit under the width row since the R2b-2 + R3 merge: the
notes states' ``right`` grew by 47 px (566: D 354 -> 401, E 1030 -> 1077;
1000: D 328 -> 375, E 991 -> 1038); ``needed`` is unchanged, the left column
still decides.  The lock line is R4's (board F).  Run
one file at a time with ``-o addopts=""``.

R6w (2026-10-07): the notes states give every predicted row its "made by"
chip -- the longest name the chip keeps (``SHORT_NAME_MAX``), one row "not
the active model" (the longest chip, two lines at the 224 px column) and one
unknown -- and the TALLEST ROW is measured (``row_max``, the delegate's own
height at the list's real width; 0 for the empty idle list): **69 -> 100
px** at both viewports (``row_max_no_chip`` = the tallest row with the
chips taken off).  Per row, at a 214 px list: a reconstructed row 51 px;
with "made by cyto3" 69 (+18, one chip line); with a 28-character name 82
(+31, the chip wraps); that name "— not the active model" 95 (+44).  ``needed`` / ``right`` / ``foot`` / ``left_fixed`` are
unchanged: the patch list scrolls.

A3 (2026-10-10, the rows v2): the left column is 240 px (minimum 240; it was
224 | 200) and the rows carry the 80 px picture, the metadata line, the voxel
line and, on a protected row, the compact tier line.  Re-pinned, each named:
``row_max`` 100 -> 123 px at both viewports (the tallest notes row, a
"not the active model" chip over an 80 px picture, in the 240 px column;
``row_max_no_chip`` 92); ``right`` of E at 566 1077 -> 1093 px (+16: the
right column is 308 px wide and its scroll viewport 300, where it was 324
and 316, so the settings wrap further).  ``needed``, ``foot`` and
``left_fixed`` are unchanged.

A3 audit fold (M2 + H1, 2026-10-10): the notes states now paint what a
finalized row really shows -- the made-by chip, the shape line, the wrapped
voxel line ("0.52 x 0.52 x 0.52 um", source in the tooltip) AND the compact
tier line of five 4-digit counts (it wraps to several lines in the 240 px
column) -- so ``row_max`` 123 -> 210 px at both viewports
(``row_max_no_chip`` 179).  Nothing else moved.

Brick B (2026-10-10, Predict + Reconstruct parity), each part named:
``left_fixed`` 260 -> 317 (D / E) and 238 -> 295 (F), so ``needed`` 423 ->
480 (D / E at both widths) and 447 -> 504 (F at 566) / 432 -> 489 (F at
1000) (audit fold M3: the line is drawn in the 10 px dim style of board D,
which took 339 / 317 / 502 / 526 / 511 down to these): the Model card's "Trained on ..." line, which WRAPS (owner fold: the
middle elision hid the channel names) and is measured in its LONGEST form --
two models on different channels, both sets named, two 44-character Step 2
run names (five lines in the 240 px column); the usual form is two lines
(+29) and the grey "not recorded" sentence in the idle states (the owner
approved the raise).  ``right`` +33 at 566 (D 401
-> 434, E 1093 -> 1126, F 401 -> 434) and +20 at 1000 (D 375 -> 395, E 1038
-> 1058, F 375 -> 395): the two-line note "Outline tolerance and cell
confidence are not used by the 3D build, so they are not shown." under the
ticks.  ``row_max`` 210 -> 197 at both viewports (``row_max_no_chip`` 179
unchanged): the "made by" chip is one line, elided at the tail, where it
wrapped to two.  ``foot`` is unchanged: the idle line with every clause
(ready, "Predict the ticked patches first.", "N protected (...) are
skipped") is shorter than the longest run line.

Why 566 (R3 audit LOW-5): 566 px is the SPEC's viewport for the 578 px
default dock (row 22) and the conservative side -- a narrower viewport
wraps more, so it measures taller.  The live app measured 572 px at the
578 px dock, 6 px wider, where the text wraps no more than here.
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
STATES = ("D", "D_notes", "E", "E_notes", "F", "F_notes")
PARTS = ("needed", "right", "foot", "left_fixed", "row_max")
TALLEST_GROUP = "gradient_descent"      # "Follow the flows" (R2a's fold)

#: measured: {viewport: {board: {part: px}}}, the max over idle and notes
CEILINGS = {
    566: {
        "D": {"needed": 480, "right": 434, "foot": 142, "left_fixed": 317,
              "row_max": 197},
        "E": {"needed": 480, "right": 1126, "foot": 142, "left_fixed": 317,
              "row_max": 197},
        "F": {"needed": 504, "right": 434, "foot": 142, "left_fixed": 295,
              "row_max": 197},
    },
    1000: {
        "D": {"needed": 480, "right": 395, "foot": 124, "left_fixed": 317,
               "row_max": 197},
        "E": {"needed": 480, "right": 1058, "foot": 124, "left_fixed": 317,
               "row_max": 197},
        "F": {"needed": 489, "right": 395, "foot": 124, "left_fixed": 295,
               "row_max": 197},
    },
}
LEFT_W = 240
RIGHT_MIN = 300
#: a real refined model's name reaches 434 px
LONG_MODEL = ("wound-v2-membrane-nuclei-refined-2026-10-04-from-step2 · "
              "round 12 (lateral)")
LONG_PATCH = "WholeScale_membrane_patch_0042_long.tiff"
LONG_RUN = "wound-v2-membrane-nuclei-refined-2026-10-04"


def _hfw(widget, width):
    if widget.hasHeightForWidth():
        return widget.heightForWidth(width)
    return widget.sizeHint().height()


def longest_status() -> str:
    """The longest sentence a run writes on the status line, from the
    shared templates Step 3 renders (``theme.PIPE_*``)."""
    from napari_epicycle.ui_shell import theme as T
    time_str = T.pipe_time("1h 02m 14s", "~2h 10m 00s remaining")
    units = T.pipe_units("patch")
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    # B: the idle line with every clause it can carry
    idle = P.S3_GATE_JOIN.join((
        P.S3_GATE_READY.format(n=12), P.S3_GATE_NOTHING_TO_RECONSTRUCT,
        P.S3_GATE_SKIPPED_PLURAL.format(k=10, statuses="accepted, finalized")))
    texts = (idle,
        T.PIPE_SEGMENTING_VIEW.format(
            unit="patch", i=12, n=40, name=LONG_PATCH, VIEW="XZ", c=412,
            t=768, time=time_str),
        T.PIPE_RECONSTRUCTING_ITEM.format(
            i=12, n=40, name=LONG_PATCH, time=time_str),
        T.PIPE_SEGMENTATION_DONE.format(s=37, n=40, units=units)
        + T.PIPE_FAILED_SUFFIX.format(k=3)
        + T.PIPE_TIME_SUFFIX.format(t="1h 02m 14s"),
    )
    return max(texts, key=len)


def measure_all(app) -> dict:
    """Runs in the subprocess."""
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
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
    content = sec._scroll.widget()
    note_texts = [t for t in (P.S3_LAST_RUN_NOTE, P.S3_LAST_RUN_VANISHED,
                              getattr(P, "S3_USEG3D_UNAVAILABLE", ""))
                  if t]
    longest_note = max(note_texts, key=len)
    status = longest_status()
    rows = {f"WholeScale_patch_{i}.tiff": s for i, s in enumerate(
        ("unprocessed", "predicted", "reconstructed", "curating", "accepted",
         "finalized", "reconstructed", "predicted", "unprocessed",
         "curating"))}
    out = {"font": app.font().family(), "status": status}

    cw_texts = (P.S3_CELL_WIDTH_PX.format(px=1000, um="0.10"),
                P.S3_NO_VOXEL_WARNING)

    lock_text = max((t for _f, t in P.S3_LOCK_SENTENCES), key=len)
    # R6: every predicted row carries its "made by" chip -- the longest
    # name the chip keeps, one row not the active model (the longest
    # chip), one unknown
    long_short = "w" * (P.SHORT_NAME_MAX - 1) + P.ELLIPSIS
    made_by = {n: (P.MADE_BY.format(short=long_short), True)
               for n, st in rows.items() if st != "unprocessed"}
    made_by["WholeScale_patch_6.tiff"] = (
        P.MADE_BY.format(short=long_short), False)
    made_by["WholeScale_patch_7.tiff"] = (P.MADE_BY_UNKNOWN, True)

    def row_heights():
        """``(tallest row, tallest row without its chip)`` at the list's
        real width (0 for an empty list)."""
        from qtpy.QtCore import QRect, Qt
        lst = w._patch_state_list
        d = lst.itemDelegate()
        width = max(lst.viewport().width(), 60)
        with_chip, without = [0], [0]
        for i in range(lst.count()):
            rec = dict(lst.item(i).data(Qt.UserRole + 2) or {})
            with_chip.append(d.layout(QRect(0, 0, width, 0), rec,
                                      lst.font())["height"])
            rec.pop("chips_top", None)
            rec.pop("info", None)
            without.append(d.layout(QRect(0, 0, width, 0), rec,
                                    lst.font())["height"])
        return max(with_chip), max(without)

    def stage(notes: bool, advanced: bool, locked: bool = False):
        # F (V1, row 21): the longest lock sentence over the run row, the
        # surface locked and the ticked lines given way to the list note
        w.set_pr_lock_line(lock_text if locked else "")
        w.set_pr_surface_locked(locked)
        if notes:
            w.show_pr_last_run_note(True, longest_note)
            w.pr_cell_width_lines.show_lines(*cw_texts)
            w.show_pr_model(LONG_MODEL, LONG_MODEL, LONG_MODEL, LONG_MODEL)
            # B: the Model card's trained line, its longest form -- two
            # models on different channels, two 44-character Step 2 runs
            join = P.S3_MODEL_CHANNEL_JOIN.join
            w.show_pr_trained(P.S3_MODEL_TRAINED_DIFFER.format(
                lat=join((P.S3_MODEL_CHANNEL.format(name="membrane", index=1),
                          P.S3_MODEL_CHANNEL.format(name="nuclei", index=0))),
                ax=join((P.S3_MODEL_CHANNEL.format(name="membrane", index=0),)),
                runs=P.S3_MODEL_TRAINED_DIFFER_RUNS.format(
                    lat=LONG_RUN, ax=LONG_RUN + "-b")), True)
            # the painted rows with a missing row and a ticked curating row
            w.refresh_patch_state_list(
                rows, missing={"WholeScale_patch_2.tiff"},
                ticked={"WholeScale_patch_3.tiff"},
                info_by_patch={n: {"labels_shape": [124, 240, 240],
                                   "num_cells_final": 4653} for n in rows},
                made_by_by_patch=made_by,
                shapes={n: (2, 124, 240, 240) for n in rows},
                spacing_by_patch={n: {"zyx": [0.52, 0.52, 0.52],
                                      "source": "step1"} for n in rows})
            # A3 audit M2: the protected rows (accepted / finalized) also
            # carry the compact tier line -- the tallest row there is
            from napari_epicycle.fusion_reconstruction import (
                _step3_tier_counts as TC)
            five = TC.tier_items({t: 1000 + i for i, t in enumerate(TC.TIERS)},
                                 {t: (100, 100, 100) for t in TC.TIERS})
            w.set_tier_counts({n: five for n in rows}, replace=True)
            pump(3)
            w.show_pr_status(status, "warn")
        else:
            w.show_pr_last_run_note(False)
            w.pr_cell_width_lines.show_lines("", "")
            w.show_pr_model("cyto3", "cyto3", "cyto3", "cyto3")
            # B: a Step 2 model with no training record -- the grey "not
            # recorded" line (a built-in model hides the line)
            w.show_pr_trained(P.S3_MODEL_TRAINED_UNKNOWN, False)
            w.refresh_patch_state_list({})
            pump(3)
            w.show_pr_status(P.S3_GATE_PICK_MODEL, "neutral")
        if advanced:
            w._pr_adv_collapsible.expand(animate=False)
            w._pr_adv_groups[TALLEST_GROUP].expand(animate=False)
        else:
            w._pr_adv_groups[TALLEST_GROUP].collapse(animate=False)
            w._pr_adv_collapsible.collapse(animate=False)
        pump(8)

    def left_fixed() -> int:
        lay = w.pr_left.layout()
        width = w.pr_left.width() - lay.contentsMargins().right()
        total, n = 0, 0
        for i in range(lay.count()):
            item = lay.itemAt(i)
            wid = item.widget()
            if wid is w._patch_state_list or (wid is not None
                                              and wid.isHidden()):
                continue
            total += (item.heightForWidth(width) if item.hasHeightForWidth()
                      else item.sizeHint().height())
            n += 1
        return total + lay.spacing() * n

    for vp in VIEWPORTS:
        for _ in range(6):           # settle the dock so the viewport is vp
            delta = vp - sec._scroll.viewport().width()
            if delta == 0:
                break
            w.resize(w.width() + delta, 2400)
            pump(5)
        got = {"viewport": sec._scroll.viewport().width()}
        for state in STATES:
            stage(notes=state.endswith("_notes"),
                  advanced=state.startswith("E"),
                  locked=state.startswith("F"))
            right_w = w.pr_right_scroll.viewport().width()
            g = {
                "section": _hfw(content, vp),
                "right": _hfw(w.pr_right_content, right_w),
                "foot": _hfw(w.pr_foot, w.pr_foot.width()),
                "left_fixed": left_fixed(),
                # the two columns at their REAL widths
                "left_col": _hfw(w.pr_left, w.pr_left.width()),
                "right_col": _hfw(w.pr_right, w.pr_right.width()),
                "section_min_w": content.minimumSizeHint().width(),
                "right_min_w": w.pr_right_content.minimumSizeHint().width(),
                "left_min_w": w.pr_left.minimumSizeHint().width(),
                "left_w": w.pr_left.width(),
                "right_w": w.pr_right.width(),
                "right_scroll_w": right_w,
            }
            g["row_max"], g["row_max_no_chip"] = row_heights()
            g["chrome"] = g["section"] - w.pr_splitter.sizeHint().height()
            g["needed"] = g["chrome"] + max(g["left_col"], g["right_col"])
            got[state] = g
        stage(notes=False, advanced=False)
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
from tests.fusion_reconstruction import test_step3_pr_layout_ceilings as t
print("S3LJSON" + json.dumps(t.measure_all(app)), flush=True)
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
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("S3LJSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:],
                   proc.stderr[-3000:])
    got = json.loads(lines[-1][len("S3LJSON"):])
    print(f"[CEIL] {json.dumps(got)}")
    return got


def test_measured_in_the_users_font(measured):
    if sys.platform == "win32" and os.path.isdir(_FONT_DIR):
        assert measured["font"] == "Segoe UI"


@pytest.mark.parametrize("vp", VIEWPORTS)
def test_the_viewport_is_the_one_asked_for(measured, vp):
    assert measured[str(vp)]["viewport"] == vp


@windows_font_metrics
@pytest.mark.parametrize("vp", VIEWPORTS)
@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("part", PARTS)
def test_no_part_is_taller_than_measured(measured, vp, state, part):
    h = measured[str(vp)][state][part]
    print(f"[CEIL] {vp} {state} {part}: {h} px")
    assert h <= CEILINGS[vp][state[0]][part]


@pytest.mark.parametrize("vp", VIEWPORTS)
@pytest.mark.parametrize("state", STATES)
def test_qts_own_section_height_fits_too(measured, vp, state):
    assert measured[str(vp)][state]["section"] <= \
        CEILINGS[vp][state[0]]["needed"]


@pytest.mark.parametrize("vp", VIEWPORTS)
def test_the_notes_cost_what_they_show(measured, vp):
    """The notes state really shows its lines (the max is not vacuous):
    the status wraps to two boxed lines, the note slot shows."""
    got = measured[str(vp)]
    for board in ("D", "E"):
        for part in ("right", "foot"):
            assert got[board + "_notes"][part] > got[board][part], \
                (board, part)


@pytest.mark.parametrize("vp", VIEWPORTS)
def test_the_chip_costs_what_it_shows(measured, vp):
    """R6w: the notes states really paint the chip (the row ceiling is not
    vacuous), and the idle list has no rows."""
    got = measured[str(vp)]
    for board in ("D", "E", "F"):
        assert got[board + "_notes"]["row_max"] >             got[board + "_notes"]["row_max_no_chip"]
        assert got[board]["row_max"] == 0


@pytest.mark.parametrize("vp", VIEWPORTS)
@pytest.mark.parametrize("state", STATES)
def test_nothing_scrolls_sideways(measured, vp, state):
    """The model of ``test_step4_panel_skeleton.py::
    test_nothing_scrolls_sideways``: every minimum width fits."""
    got = measured[str(vp)][state]
    assert got["section_min_w"] <= vp
    assert got["right_min_w"] <= got["right_scroll_w"]
    assert got["left_min_w"] <= LEFT_W


@pytest.mark.parametrize("vp", VIEWPORTS)
@pytest.mark.parametrize("state", STATES)
def test_the_columns_keep_steps_4s_numbers(measured, vp, state):
    got = measured[str(vp)][state]
    assert got["left_w"] == LEFT_W          # stretch 0: the left stays put
    assert got["right_w"] >= RIGHT_MIN


def test_the_settings_scroll_inside_the_right_column(measured):
    """Advanced open grows the right column's CONTENT, not the section."""
    for vp in VIEWPORTS:
        got = measured[str(vp)]
        assert got["E"]["right"] > got["D"]["right"]
        assert got["E"]["needed"] == got["D"]["needed"]

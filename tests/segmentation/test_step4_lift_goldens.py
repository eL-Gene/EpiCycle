"""The three goldens of the R1 lift.

Captured ONCE from the untouched Step 4 code, before anything moved into
``ui_shell/``, and never regenerated to make a test pass:

* ``golden/step4_panel_tree.json`` -- the whole ``SegmentStack`` widget tree
  and the Export window (no workspace, and a workspace with stacks);
* ``golden/step4_row_delegate.json`` -- ``sizeHint``, a painted-image hash
  and the tooltip per painted part, for fixed records in the three modes,
  taken in a SUBPROCESS with the real font;
* ``golden/step4_text_names.json`` -- every name of ``_step4_text`` and
  ``_step4_models`` with its value, compared as dicts by EQUALITY.

A golden is written only with ``STEP4_LIFT_GOLDEN_WRITE=1`` and only when
its file is missing.  Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import _step4_goldens as G

_REPO = Path(__file__).resolve().parents[2]
_FONT_DIR = "C:/Windows/Fonts"
_WRITE = os.environ.get("STEP4_LIFT_GOLDEN_WRITE") == "1"


def _load_or_write(path: Path, got):
    if not path.exists():
        if not _WRITE:
            pytest.fail(f"golden {path.name} is missing; capture it from the "
                        "untouched code with STEP4_LIFT_GOLDEN_WRITE=1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=1, ensure_ascii=False,
                                   sort_keys=False) + "\n", encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def _first_difference(want, got, where=""):
    if type(want) is not type(got):
        return f"{where}: {want!r} != {got!r}"
    if isinstance(want, dict):
        for k in want:
            if k not in got:
                return f"{where}/{k}: missing"
            d = _first_difference(want[k], got[k], f"{where}/{k}")
            if d:
                return d
        extra = [k for k in got if k not in want]
        return f"{where}: extra keys {extra}" if extra else ""
    if isinstance(want, list):
        if len(want) != len(got):
            return f"{where}: length {len(want)} != {len(got)}"
        for i, (a, b) in enumerate(zip(want, got)):
            d = _first_difference(a, b, f"{where}[{i}]")
            if d:
                return d
        return ""
    return "" if want == got else f"{where}: {want!r} != {got!r}"


# ---------------------------------------------------------------- the panel

def test_the_panel_tree_is_the_golden(qapp, tmp_path):
    got = json.loads(json.dumps(G.build_panel_states(qapp, tmp_path)))
    want = _load_or_write(G.PANEL_GOLDEN, got)
    assert want == got, _first_difference(want, got)


def test_the_panel_golden_has_teeth():
    """A changed word anywhere in the tree is a difference."""
    want = json.loads(G.PANEL_GOLDEN.read_text(encoding="utf-8"))
    tree = want["workspace"]["tree"]
    assert len(tree) > 300
    assert any(e.get("collapsible") for e in tree)
    assert any(e.get("windowTitle") for e in tree)       # the Export window
    changed = json.loads(json.dumps(want))
    texts = [e for e in changed["workspace"]["tree"] if e.get("text")]
    texts[0]["text"] += "!"
    assert _first_difference(want, changed)


# ---------------------------------------------------------------- the rows

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
from tests.segmentation import _step4_goldens as G
print("S4JSON" + json.dumps({"font": app.font().family(),
                             "rows": G.dump_delegate()}), flush=True)
os._exit(0)
"""


@pytest.fixture(scope="module")
def painted():
    code = f"FONT_DIR = {_FONT_DIR!r}\n" + _SUBPROCESS
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("S4JSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:],
                   proc.stderr[-3000:])
    return json.loads(lines[-1][len("S4JSON"):])


def test_the_rows_paint_as_the_golden(painted):
    if sys.platform == "win32" and os.path.isdir(_FONT_DIR):
        assert painted["font"] == "Segoe UI"
    got = painted["rows"]
    want = _load_or_write(G.DELEGATE_GOLDEN, got)
    assert want == got, _first_difference(want, got)


def test_the_row_golden_covers_every_mode_and_part():
    want = json.loads(G.DELEGATE_GOLDEN.read_text(encoding="utf-8"))
    modes = {k.split("/")[0] for k in want}
    assert modes == {"wide", "narrow", "measure"}
    tips = {t for v in want.values() for t in v["tips"].values()}
    assert len(tips) >= 4        # tick, eye, chip, ③'s click, and "none"
    assert len({v["sha1"] for v in want.values()}) > 30


# ---------------------------------------------------------------- the text

def test_the_text_tables_are_the_golden():
    got = G.dump_text_names()
    want = _load_or_write(G.TEXT_GOLDEN, got)
    for mod in ("_step4_text", "_step4_models"):
        missing = [k for k in want[mod] if k not in got[mod]]
        assert not missing, (mod, missing)
        changed = [k for k in want[mod] if want[mod][k] != got[mod][k]]
        assert not changed, (mod, changed)


def test_the_text_golden_compares_dicts_by_equality():
    want = json.loads(G.TEXT_GOLDEN.read_text(encoding="utf-8"))
    lines = want["_step4_text"]["STEP4_HELP_LINES"]
    assert isinstance(lines, dict) and len(lines) > 80
    reordered = dict(reversed(list(lines.items())))
    assert reordered == lines            # order is not a difference
    assert dict(lines, rescale_cb="x") != lines      # a word is


# ---------------------------------------------------------------- the layout

_LAYOUT_WRITE = os.environ.get("STEP4_LAYOUT_GOLDEN_WRITE") == "1"

_LAYOUT_SUBPROCESS = """
import json, os, sys, tempfile
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
import napari_epicycle  # the import order the app uses
from tests.segmentation import _step4_goldens as G
tmp = tempfile.mkdtemp(prefix="s4layout_")
print("S4JSON" + json.dumps({"font": app.font().family(),
                             "layout": G.build_panel_layout_states(app, tmp)}),
      flush=True)
os._exit(0)
"""


@pytest.fixture(scope="module")
def laid_out():
    code = f"FONT_DIR = {_FONT_DIR!r}\n" + _LAYOUT_SUBPROCESS
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("S4JSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:],
                   proc.stderr[-3000:])
    return json.loads(lines[-1][len("S4JSON"):])


def test_the_panel_layout_is_the_golden(laid_out):
    """Layout and interaction properties the tree golden does not hold:
    size policies, set minimum / maximum sizes, cursors, mouse
    transparency, a label's wordWrap / textFormat / alignment / buddy, and
    every layout's margins, spacing, grid stretches and items IN ORDER.

    ``golden/step4_panel_layout.json`` was captured from the PRE-LIFT
    commit ``db8a0917`` (a detached worktree, this capture code copied in,
    run twice and diffed for determinism) in this real-font subprocess,
    and is never regenerated to make the test pass; it is written only
    with ``STEP4_LAYOUT_GOLDEN_WRITE=1`` AND the file missing."""
    if sys.platform == "win32" and os.path.isdir(_FONT_DIR):
        assert laid_out["font"] == "Segoe UI"
    got = laid_out["layout"]
    path = G.LAYOUT_GOLDEN
    if not path.exists():
        if not _LAYOUT_WRITE:
            pytest.fail(f"golden {path.name} is missing; capture it from the "
                        "pre-lift code with STEP4_LAYOUT_GOLDEN_WRITE=1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    want = json.loads(path.read_text(encoding="utf-8"))
    assert want == got, _first_difference(want, got)


def test_the_layout_golden_covers_what_the_audit_asked():
    want = json.loads(G.LAYOUT_GOLDEN.read_text(encoding="utf-8"))
    tree = want["workspace"]["tree"]
    assert len(tree) > 300
    labels = [e["label"] for e in tree if "label" in e]
    assert any(lb["wordWrap"] for lb in labels)
    assert any(lb["buddy"] for lb in labels)              # adv_row's ticks
    assert any("cursor" in e for e in tree)               # TickLabel
    assert any(e.get("transparentForMouse") for e in tree)  # engine names
    assert any("maximum" in e for e in tree)              # field(): 128 px
    nested = []

    def walk(lay):
        nested.append(lay)
        for it in lay["items"]:
            if "layout" in it:
                walk(it["layout"])
    for e in tree:
        if "layout" in e:
            walk(e["layout"])
    assert any(lay["class"] == "QGridLayout" and lay["spacing"] == [8, 5]
               for lay in nested)                          # frm4
    changed = json.loads(json.dumps(want))
    first = next(e for e in changed["workspace"]["tree"] if "label" in e)
    first["label"]["wordWrap"] = not first["label"]["wordWrap"]
    assert _first_difference(want, changed)


# ---------------------------------------------------------------- Set spacing

_SPACING_WRITE = os.environ.get("STEP4_SPACING_GOLDEN_WRITE") == "1"


def test_the_spacing_dialog_is_the_golden(qapp):
    """Step 4's Set spacing… dialog (R7a): its widget tree, layout tree and ``values()`` in three states.

    ``golden/step4_spacing_dialog.json`` was captured from the UNTOUCHED
    ``segmentation/_set_spacing_dialog.py`` (branch
    ``feature/step3-recon-r7a`` at ``326fd5a3``, before the lift into
    ``ui_shell/_set_spacing_dialog.py``) and is never regenerated to make
    the test pass; it is written only with ``STEP4_SPACING_GOLDEN_WRITE=1``
    AND the file missing."""
    got = G.build_spacing_dialog_states(qapp)
    path = G.SPACING_DIALOG_GOLDEN
    if not path.exists():
        if not _SPACING_WRITE:
            pytest.fail(f"golden {path.name} is missing; capture it from the "
                        "pre-lift code with STEP4_SPACING_GOLDEN_WRITE=1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    want = json.loads(path.read_text(encoding="utf-8"))
    assert want == got, _first_difference(want, got)


def test_the_spacing_golden_has_teeth():
    want = json.loads(G.SPACING_DIALOG_GOLDEN.read_text(encoding="utf-8"))
    assert set(want) == {"one_user", "three_none", "one_bad"}
    one = want["one_user"]
    assert one["values"] == {"zyx": [2.0, 0.5, 0.25],
                             "z_orientation": "basal_up",
                             "axes_override": "ZCYX"}
    assert want["one_bad"]["values"]["zyx"][0] == 1.0
    classes = [e["class"] for e in one["tree"]]
    assert classes[0] == "SetSpacingDialog"
    assert classes.count("QDoubleSpinBox") == 3
    assert classes.count("QComboBox") == 2
    assert any(e.get("text") == "OK" for e in one["tree"])
    assert any("layout" in e for e in one["layout"])
    changed = json.loads(json.dumps(want))
    next(e for e in changed["one_user"]["tree"]
         if e.get("text"))["text"] += "!"
    assert _first_difference(want, changed)

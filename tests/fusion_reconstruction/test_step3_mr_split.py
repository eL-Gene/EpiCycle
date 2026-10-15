"""Model Refinement split brick: the right column's vertical splitter
(settings over the foot), the Loss plot filling its pane, the open / close
sizing and the user's drag.  Runs in a subprocess (offscreen Qt)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FONT_DIR = "C:/Windows/Fonts"

_SUBPROCESS = """
import json, os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication, QSplitter, QWidget
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
import napari_epicycle
from napari_epicycle.fusion_reconstruction.ui import (
    FusionReconstructionCuration)
from napari_epicycle.ui_shell.theme import SPLITTER_STYLE

def pump(n=8):
    for _ in range(n):
        app.processEvents()

def build(width):
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    w._accordion.setCurrentIndex(3)
    w.resize(width, 1100)
    w.show()
    pump(12)
    return w

def snap(w):
    sp = w.refine_right_splitter
    return {"sizes": sp.sizes(), "foot_h": w.refine_foot.height(),
            "plot_h": w.refinement_loss_plot.height(),
            "plot_w": w.refinement_loss_plot.width(),
            "loss_h": w.refine_loss_collapsible.height(),
            "total": sp.height()}

out = {}
for tag, width in (("a", 700), ("b", 1400)):
    w = build(width)
    sp = w.refine_right_splitter
    r = {}
    r["is_splitter"] = isinstance(sp, QSplitter)
    r["vertical"] = sp.orientation() == Qt.Vertical
    r["count"] = sp.count()
    r["children"] = [sp.widget(0) is w.refine_right_scroll,
                     sp.widget(1) is w.refine_foot]
    r["style"] = sp.styleSheet() == SPLITTER_STYLE
    r["handle"] = sp.handleWidth()
    r["collapsible"] = sp.childrenCollapsible()
    r["btn_visible"] = w.launch_refinement_btn.isVisible()
    r["closed0"] = snap(w)
    w.refine_loss_collapsible.expand(animate=False)
    pump(12)
    r["open1"] = snap(w)
    r["foot_chrome"] = (w.refine_foot.height()
        - w.refinement_loss_plot.height())
    w.refine_loss_collapsible.collapse(animate=False)
    pump(12)
    r["closed1"] = snap(w)
    # a drag, then open / close keep the user's sizes
    total = sp.height() - sp.handleWidth()
    sp.setSizes([total - 300, 300])
    sp.splitterMoved.emit(total - 300, 1)
    pump(8)
    r["dragged"] = sp.sizes()
    w.refine_loss_collapsible.expand(animate=False)
    pump(12)
    r["drag_open"] = sp.sizes()
    w.refine_loss_collapsible.collapse(animate=False)
    pump(12)
    r["drag_closed"] = sp.sizes()
    out[tag] = r
    w.close()
print("S3SPLITJSON" + json.dumps(out), flush=True)
os._exit(0)
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
             if ln.startswith("S3SPLITJSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:], proc.stderr[-3000:])
    got = json.loads(lines[-1][len("S3SPLITJSON"):])
    print(f"[SPLIT] {json.dumps(got)}")
    return got


@pytest.fixture(params=("a", "b"))
def m(request, measured):
    return measured[request.param]


def test_the_splitter_has_the_two_children(m):
    assert m["is_splitter"] and m["vertical"] and m["count"] == 2
    assert m["children"] == [True, True]
    assert m["style"] and m["handle"] == 6
    assert m["collapsible"] is False


def test_open_gives_the_foot_at_least_half(m):
    top, bottom = m["open1"]["sizes"]
    assert bottom >= 0.5 * (top + bottom), m["open1"]


def test_the_plot_fills_its_pane(m):
    o = m["open1"]
    chrome = m["foot_chrome"]
    assert o["plot_h"] >= o["foot_h"] - chrome - 8, (o, chrome)
    # chrome (buttons + bar + status + header + legend) stays modest
    assert chrome < 200, chrome


def test_close_returns_the_collapsed_height(m):
    c0, c1 = m["closed0"], m["closed1"]
    assert abs(c1["sizes"][1] - c0["sizes"][1]) <= 2, (c0, c1)
    assert c1["sizes"][1] >= m["closed0"]["foot_h"] - 2
    assert c1["sizes"][0] > c1["sizes"][1]


def test_a_drag_is_respected_on_open_and_close(m):
    assert m["drag_open"] == m["dragged"], m
    assert m["drag_closed"] == m["dragged"], m


def test_the_buttons_stay_visible(m):
    assert m["btn_visible"]

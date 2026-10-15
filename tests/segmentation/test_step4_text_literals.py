"""I-18 widened to every Step 4 site (B2 fix pass R1-6): no string literal, f-string or ``"..." + x`` reaches a
``setText`` / ``setToolTip`` / status push in ``ui.py``, ``controller.py``,
``analysis_widget.py`` or ``_export_dialog.py`` -- every sentence on screen
comes from a named constant.  AST-only.

``test_step4_status_literals.py`` keeps its narrower B1a / B1b checks; this
file is the whole-module sweep.  Keyword arguments count too
(``set_busy(..., reason=...)``, ``show_status(..., tip=...)``).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SEG = Path(__file__).resolve().parents[2] / "napari_epicycle" / "segmentation"

TEXT_CALLS = {"setText", "setToolTip", "setPlaceholderText",
              "setWindowTitle"}
#: every Step 4 status push (a sentence reaches a label through these)
PUSHES = {"show_seg_status", "show_mt_status", "show_finalize_note",
          "show_not_reconstructed", "set_measure_button_text", "show_status",
          "show_last", "set_lock_lines", "set_busy", "show_card_message",
          "show_map_result", "show_map_error", "show_groups_result",
          "set_mt_unsaved", "show_stack_card", "set_spacing_note"}
TEXT_KEYWORDS = {"reason", "tip", "text", "line", "note", "pill"}
#: pushes whose SECOND positional argument is a tooltip, not a kind (LOW-4)
TIP_SECOND = {"show_mt_status", "show_seg_status", "show_status"}
#: the Workbench's rich-text label helpers (LOW-3): their words are the
#: fourteen form labels and five engine names of the mockup, nothing else
MARKUP_HELPERS = {"_heading", "_form_label"}
FORM_LABELS = {
    "Method", "Components", "Scaling", "Clip extremes",
    "Remove the effect of", "Neighbours", "Spacing", "Start from",
    "Distance", "Perplexity", "Resolution", "Smallest group",
    "Number of groups", "Seed"}
ENGINE_EXTRAS = {"winsorize", "regress out", "min_dist", "kNN", "k",
                 "min_size"}

#: Functions left as they are, and why (each one is a documented
#: exception, not an oversight):
EXEMPT = {
    # Diagnostics as today: the GPU line is a runtime probe result
    ("ui.py", "_detect_gpu"),
    ("ui.py", "_build_diagnostics_panel"),
    # rich-text MARKUP around a caller's constant (the words are the
    # caller's; the f-string only adds the span styles)
    ("analysis_widget.py", "_heading"),
    ("analysis_widget.py", "_form_label"),
}


def _bad(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip() != ""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp):
        return _bad(node.left) or _bad(node.right)
    if isinstance(node, ast.IfExp):          # MED-1: both branches
        return _bad(node.body) or _bad(node.orelse)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)):
        return True                           # MED-1: "…".format(…)
    return False


def _markup_offences(tree):
    """LOW-3: every _heading / _form_label call site passes a constant or
    one of the allowed label words."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (
            f.id if isinstance(f, ast.Name) else "")
        if name not in MARKUP_HELPERS:
            continue
        vals = list(node.args[:2] if name == "_heading" else node.args[:1])
        vals += [k.value for k in node.keywords if k.arg == "extra"]
        for v in vals:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                if v.value not in FORM_LABELS | ENGINE_EXTRAS:
                    out.append((node.lineno, v.value))
            elif _bad(v):
                out.append((node.lineno, ast.unparse(v)[:60]))
    return out


def offences(source: str, module: str):
    tree = ast.parse(source)
    out = []
    parents = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                parents.setdefault(id(n), fn.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        if f.attr not in TEXT_CALLS | PUSHES:
            continue
        if (module, parents.get(id(node), "")) in EXEMPT:
            continue
        # the sentence is the first argument (a push's second positional
        # argument is its kind, "warn" / "ok"); keywords by name
        args = list(node.args)[:2 if f.attr in TIP_SECOND else 1]
        kw = [k.value for k in node.keywords if k.arg in TEXT_KEYWORDS]
        if any(_bad(a) for a in args + kw):
            out.append((node.lineno, ast.unparse(node)[:100]))
    return out


#: the modules Step 4 moved into ui_shell/ are read where they live now, never through their alias
MOVED = ["ui_shell/_stack_row_delegate.py",
         "ui_shell/_recon_settings_widgets.py",
         "ui_shell/_stack_thumbs.py"]
#: the shared Set spacing dialog Step 4's module SUBCLASSES since R7a
#: : checked where its code lives (it is
#: not an alias, so it is not in ``MOVED``)
LIFTED = ["ui_shell/_set_spacing_dialog.py"]
MODULES = ["ui.py", "controller.py", "analysis_widget.py",
           "_export_dialog.py", "_set_spacing_dialog.py"] + MOVED + LIFTED


def _path(module: str) -> Path:
    return (SEG.parent / module) if "/" in module else (SEG / module)


@pytest.mark.parametrize("module", MODULES)
def test_no_literal_reaches_a_step4_label(module):
    bad = offences(_path(module).read_text(encoding="utf-8"), module)
    assert not bad, bad


def test_the_markup_helpers_carry_only_the_form_labels():
    tree = ast.parse((SEG / "analysis_widget.py").read_text(
        encoding="utf-8"))
    bad = _markup_offences(tree)
    assert not bad, bad


def test_the_new_rules_have_teeth():
    src = ('def f(self):\n'
           '    x.setText(a if a else "No description.")\n'
           '    x.setToolTip("n={n}".format(n=1))\n'
           '    self.ui.show_mt_status(T.A, "a raw tip")\n'
           '    self.ui.show_mt_status(T.A, T.B)\n')
    assert len(offences(src, "x.py")) == 3
    tree = ast.parse('_form_label("Seed", p)\n_form_label("Seed now!", p)\n'
                     '_heading(T.A, "desc", p)\n')
    assert len(_markup_offences(tree)) == 2


def test_the_check_has_teeth():
    src = ('def f(self):\n'
           '    self.ui.x.setText("Idle")\n'
           '    self.ui.set_lock_lines(f"x {y}")\n'
           '    dlg.set_busy(True, reason="Busy")\n'
           '    dlg.set_busy(True, reason=T.EXPORT_RUNNING)\n'
           '    self.ui.x.setText(T.STEP4_SEG_IDLE)\n'
           '    aw.show_card_message("", "warn")\n')
    assert len(offences(src, "x.py")) == 3


def test_the_controller_builds_no_widget_with_literal_text():
    """LOW-5: the controller's own dialogs (the merged-columns question of
    Export features) build their QLabel / QCheckBox / QPushButton / button
    texts from constants."""
    tree = ast.parse((SEG / "controller.py").read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else "")
        if name in ("QLabel", "QCheckBox", "QPushButton", "addButton") \
                and _bad(node.args[0]):
            bad.append((node.lineno, ast.unparse(node)[:90]))
    assert not bad, bad


def test_the_moved_modules_are_read_where_they_live():
    """The moved modules are checked
    in ``ui_shell/`` -- the old ``segmentation/`` path is a three-line alias
    an AST check would pass vacuously."""
    for module in MOVED:
        real = _path(module)
        assert real.parent.name == "ui_shell", module
        assert len(real.read_text(encoding="utf-8").splitlines()) > 100
        old = SEG / real.name
        if old.exists():
            assert "sys.modules[__name__]" in old.read_text(encoding="utf-8")

"""No literal text reaches a ① / ② / ③ / Export-window label
(B1a scope, widened by B1b to ③ and the Export
window; B2's ``test_step4_text_literals.py`` widens it to every Step 4
site).

AST-only.  Checked:
* in the B1a modules (``_stack_row_delegate``, ``_set_spacing_dialog``,
  ``_step4_rows``) every ``setText`` / ``setToolTip`` / ``addItem`` /
  ``addAction`` argument;
* in ``ui.py``, every text a ① / ② builder or setter puts on a widget
  (constructors ``QLabel`` / ``QPushButton`` / ``QCheckBox`` included);
* in ``controller.py``, every ``setText`` / ``show_seg_status`` on the ① / ②
  status labels.
An argument is fine when it is a name, an attribute, a call or ``""``; a
string literal, an f-string or a ``"..." + x`` is not.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SEG = Path(__file__).resolve().parents[2] / "napari_epicycle" / "segmentation"

TEXT_CALLS = {"setText", "setToolTip", "addItem", "addItems", "addAction",
              "setPlaceholderText", "setSpecialValueText", "setSuffix"}
CTORS = {"QLabel", "QPushButton", "QCheckBox", "QRadioButton"}
#: ① / ② status labels the controller writes
STATUS = {"run_status_label", "session_stacks_status", "seg_ticked_label",
          "model_active_indicator_label", "model_trained_label",
          "run_row_note", "set_spacing_note", "session_ticked_label",
          "cell_width_px_label", "session_lock_line", "seg_lock_line",
          "mt_lock_line",
          # ③ and the Export window (B1b)
          "qc_status_line", "finalize_note", "stack_card_name",
          "stack_card_pill", "stack_card_line", "stack_card_note",
          "mt_chip_measure", "mt_chip_review", "mt_chip_tidy",
          "mt_not_recon_label", "export_status_label", "count_label",
          "mesh_target_label", "mesh_what_label", "last_label"}
#: ③ / Export panel setters the controller calls with a sentence first
SETTERS = {"show_seg_status", "show_mt_status", "show_finalize_note",
           "show_not_reconstructed", "set_measure_button_text",
           "show_status", "show_last",
           # B2 (R1-6): the lock lines and the Export window's busy reason
           "set_lock_lines", "set_busy"}
#: keyword arguments that carry a sentence (``set_busy(..., reason=)``)
TEXT_KEYWORDS = {"reason", "tip"}
#: ui.py functions left as they are (Diagnostics as today)
UI_NOT_B1A = {"_detect_gpu", "_build_diagnostics_panel"}


def _bad_arg(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value != "" and node.value.strip() != ""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp):
        return _bad_arg(node.left) or _bad_arg(node.right)
    return False


def _offences(tree, *, only_status=False, ctors=True):
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kw_bad = any(_bad_arg(k.value) for k in node.keywords
                     if k.arg in TEXT_KEYWORDS)
        f0 = node.func
        if kw_bad and isinstance(f0, ast.Attribute) and f0.attr in SETTERS:
            out.append((node.lineno, ast.unparse(node)[:90]))
            continue
        if not node.args:
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in TEXT_CALLS | SETTERS:
            if only_status:
                owner = f.value
                name = owner.attr if isinstance(owner, ast.Attribute) else ""
                if f.attr not in SETTERS and name not in STATUS:
                    continue
            if _bad_arg(node.args[0]):
                out.append((node.lineno, ast.unparse(node)[:90]))
        elif ctors and isinstance(f, ast.Name) and f.id in CTORS:
            if _bad_arg(node.args[0]):
                out.append((node.lineno, ast.unparse(node)[:90]))
    return out


#: the modules Step 4 moved into ui_shell/
#: are read where they live now, never through their alias
MOVED = ["ui_shell/_stack_row_delegate.py",
         "ui_shell/_recon_settings_widgets.py",
         "ui_shell/_stack_thumbs.py"]


def _path(module: str) -> Path:
    return (SEG.parent / module) if "/" in module else (SEG / module)


#: the shared Set spacing dialog Step 4's module SUBCLASSES since R7a
#:: checked where its code lives (it is
#: not an alias, so it is not in ``MOVED``)
LIFTED = ["ui_shell/_set_spacing_dialog.py"]


@pytest.mark.parametrize("module", MOVED + LIFTED + ["_set_spacing_dialog.py",
                                            "_step4_rows.py",
                                            "_export_dialog.py"])
def test_the_b1a_modules(module):
    tree = ast.parse(_path(module).read_text(encoding="utf-8"))
    assert not _offences(tree), _offences(tree)


def test_the_panel_builders_and_setters():
    tree = ast.parse((SEG / "ui.py").read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in UI_NOT_B1A:
            bad += [(node.name,) + o for o in _offences(node)]
    assert not bad, bad


def test_the_controller_writes_constants_to_the_status_labels():
    tree = ast.parse((SEG / "controller.py").read_text(encoding="utf-8"))
    bad = _offences(tree, only_status=True, ctors=False)
    assert not bad, bad


def test_the_measure_panel_is_checked():
    """③'s builder and setters are no longer excluded (B1b)."""
    tree = ast.parse((SEG / "ui.py").read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)}
    assert "_build_measure_panel" in names
    assert "_build_qc_analysis_panel" not in names
    assert not {"_build_measure_panel", "show_stack_card",
                "show_mt_chips"} & UI_NOT_B1A


def test_the_check_has_teeth():
    tree = ast.parse('self.ui.run_status_label.setText("Idle")\n'
                     'self.ui.run_status_label.setText(f"x {y}")\n'
                     'self.ui.run_status_label.setText(T.STEP4_SEG_IDLE)\n'
                     'QLabel("Workspace")\n')
    assert len(_offences(tree, only_status=True, ctors=False)) == 2
    assert len(_offences(tree)) == 3
    kw = ast.parse('dlg.set_busy(True, reason="Busy")\n'
                   'dlg.set_busy(True, reason=T.EXPORT_RUNNING)\n'
                   'self.ui.set_lock_lines("Locked")\n')
    assert len(_offences(kw, only_status=True, ctors=False)) == 2


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

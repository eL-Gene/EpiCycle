"""No string literal reaches the screen from the code R2a and R3 built
(Step 3's own words).

Modelled on ``tests/segmentation/test_step4_text_literals.py``, scoped to
the functions R2a added or rebuilt in ``fusion_reconstruction/ui.py`` and
``controller.py`` (V2 adds Session's and Curation's painted rows and the
pictures) (the rest of those two files is older code with its own
words; R3-R7 widen the scope as they rebuild it).  A literal, an f-string,
``"..." + x``, either branch of ``a if c else b`` and ``"...".format(...)``
all count, whether they reach ``setText`` / ``setToolTip`` / a widget's
constructor / a combo item / a spin box's special text or suffix / the log.
AST only.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

FR = Path(__file__).resolve().parents[2] / "napari_epicycle" / \
    "fusion_reconstruction"

#: the functions R2a added or rebuilt, per module
SCOPE = {
    "ui.py": (
        "_build_pr_settings", "_pr_adv_row", "_build_pr_hidden_knobs",
        "_build_pr_advanced", "_update_pr_flow_controls",
        "_update_pr_size_off_note", "_make_pr_group_widgets",
        "_build_pr_hidden_recon", "_apply_pr_help_lines",
        "show_pr_last_run_note", "set_pr_settings_desc",
        # R3: the section's layout, the two cards, the foot, Session's line
        "_create_predict_reconstruct_panel", "_build_pr_model_card",
        "_build_pr_patches_card", "_build_pr_foot",
        "_apply_pr_layout_help_lines", "show_pr_model",
        "set_pr_ticked_lines", "show_pr_status",
        "_build_session_model_pointer",
        # V1: the painted rows, the lock, the missing line
        "refresh_patch_state_list", "refresh_curation_bucket",
        "set_pr_lock_line", "set_pr_surface_locked", "_apply_pr_tick_lock",
        "show_pr_trained",
        "_pr_setting_inputs",
        # V2: Session's and Curation's painted rows and their pictures
        "update_patch_list_status", "_fit_session_list_height",
        "_thumb_lists", "_apply_patch_thumb", "_apply_patch_thumbs",
        "set_patch_thumbs", "set_patch_thumb",
        "_build_session_patch_header", "_build_session_patch_list",
        "_build_curation_bucket",
        # R7w: Session's Set spacing… row and its gate
        "_build_session_spacing_row", "set_session_spacing_gate",
    ),
    "controller.py": (
        "_apply_recon_defaults", "_on_epicycle_defaults_clicked",
        "_last_run_manifest", "_refresh_start_from_items",
        "_refresh_last_run_note", "_start_from_key", "_select_start_from",
        "_show_custom", "_on_start_from_chosen", "_preset_values",
        "_apply_start_from", "_apply_last_run_channels",
        "_connect_pr_setting_edits", "_on_pr_setting_edited",
        "_restore_start_from", "_refuse_start_from",
        "_collect_seg_config",
        "_collect_recon_config", "_newest_recon_manifest",
        # R2c
        "_recorded_inference", "_recon_recorded_seg",
        "_persist_prediction_cache",
        # R2b part 2
        "_patch_spacing_record", "_step1_state_for_spacing",
        "_patch_spacing", "_cell_width_patches", "_on_cell_width_changed",
        "_update_cell_width_line", "_update_cell_width_line_body",
        "_predict_single_patch",
        # R3: the one status setter and the foot's gates
        "_pr_status", "_pr_done_kind", "_show_pr_model_lines",
        "_on_pr_selection_changed",
        "_refresh_pr_foot_gates",
        # V1: the ticks, the missing files, Forget, the lock line
        "_ticked_patch_names", "_pr_stored_ticks", "_pr_runnable",
        "_pr_set_item_ticked", "_on_pr_item_changed", "_on_pr_ticks_changed",
        "_pr_tick_rows", "_on_pr_tick_all_unprocessed", "_on_pr_tick_every",
        "_recompute_missing_patches", "_pr_check_files", "_pr_row_menu",
        "_pr_forget_refusal", "_forget_patch", "_pr_forget_reset_view",
        "_refresh_pr_lock", "_on_pr_lock_timer", "_pr_row_manifests",
        # R3 audit LOW-2: the Session click writes the P+R foot
        "_on_patch_selected",
        # V2: Session's rows and the pictures
        "_refresh_patch_list_colors", "_session_axes", "_session_shapes",
        "_s3_thumb_names", "_s3_thumb_axes_for", "_s3_thumb_cache_dir",
        "_start_step3_thumbs", "_on_thumb_ready", "_on_thumb_failed",
        "_stop_step3_thumbs", "_sweep_unlisted_thumbs",
        # R6w: the model provenance -- the block, the chip, the window
        "_recon_provenance_extra", "_pr_row_made_by",
        "_pr_prediction_block", "_pr_manifest_paths",
        # B: the Model card's trained-channels line
        "_show_pr_trained_line",
        "_live_provenance_windows", "_on_pr_info_requested",
        "_save_provenance_notes", "_set_provenance_windows_locked",
        "_close_provenance_windows", "_refresh_patch_state_list",
        # R7w: the per-patch voxel size -- resolution, the readers, the
        # workers' results, Set spacing… and its menu, the gate
        "_patch_stack_path", "_spacing_main_thread_blind",
        "_patch_own_spacing", "_qc_fallback_spacing", "_spacing_zyx",
        "_resolve_listed_spacing", "_apply_worker_spacing",
        "_session_spacing", "_session_highlighted", "_spacing_locked",
        "_refresh_session_spacing_gate", "_on_set_spacing",
        "_on_session_list_menu", "_on_session_row_menu",
        "_session_row_menu", "_make_set_spacing_dialog",
        "_spacing_stale_names", "_open_set_spacing", "_on_qc_spacing_found",
        "_extraction_spacing_used", "_record_extraction_spacing",
        "_write_qc_extraction_context", "_check_qc_extraction_staleness",
        # (not _on_serial_pred_patch_done: older code with its own words)
    ),
    # R2b part 2: the cell width's two lines (kept out of ui.py)
    "_step3_cell_width.py": None,
    # R2c: the prediction manifest (Qt-free; it shows nothing, so any
    # literal reaching a screen call there is a bug)
    "_step3_manifests.py": None,
    # V1: the rows' records and Forget (Qt-free)
    "_step3_rows.py": None,
    "_step3_forget.py": None,
    # V2: the pictures' engine (V2a; its words now in _step3_pr_text)
    "_step3_thumbs.py": None,
    # R6: the model provenance (Qt-free) and its details window
    "_model_provenance.py": None,
    "_model_provenance_dialog.py": None,
}

#: method calls whose FIRST argument is a sentence on screen
TEXT_CALLS = {"setText", "setToolTip", "setPlaceholderText",
              "setWindowTitle", "setSpecialValueText", "setSuffix",
              "setPrefix", "addItem", "_append_log", "show_pr_status",
              "setTitle", "setStatusTip", "_pr_status",
              # R7w: a menu item's text
              "addAction"}
#: calls whose title AND message are the second and third arguments
#: (``QMessageBox.warning(parent, title, text, ...)``)
MESSAGE_BOX = {"warning", "information", "critical", "question"}
#: constructors whose first argument is the widget's text
CTORS = {"QLabel", "QCheckBox", "QPushButton", "QGroupBox", "collapsible"}
#: calls whose SECOND argument is a title on screen
SECOND_ARG = {"_make_subsection_header", "collapsible", "sub_header"}
#: R7w: ``themed_confirm(parent, title, body, confirm_text=...)``
CONFIRM = {"themed_confirm"}


def _bad(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip() != ""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp):
        return _bad(node.left) or _bad(node.right)
    if isinstance(node, ast.IfExp):
        return _bad(node.body) or _bad(node.orelse)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and _bad(node.func.value)):
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return any(_bad(e) for e in node.elts)
    return False


def _name(f) -> str:
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def offences(source: str, functions) -> list:
    tree = ast.parse(source)
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if functions is not None and fn.name not in functions:
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = _name(node.func)
            args = []
            if name in TEXT_CALLS | CTORS and node.args:
                args.append(node.args[0])
            if name == "addItems" and node.args:
                args.append(node.args[0])
            if name in SECOND_ARG and len(node.args) > 1:
                args.append(node.args[1])
            if name == "setItemText" and len(node.args) > 1:
                args.append(node.args[1])
            if name in MESSAGE_BOX:
                args.extend(node.args[1:3])
            if name in CONFIRM:
                args.extend(node.args[1:3])
                args.extend(k.value for k in node.keywords
                            if k.arg == "confirm_text")
            if name == "_make_subsection_header" and node.args:
                args.append(node.args[1] if len(node.args) > 1
                            else node.args[0])
            if any(_bad(a) for a in args):
                out.append((fn.name, node.lineno, ast.unparse(node)[:90]))
    return out


def _defined(source: str) -> set:
    return {n.name for n in ast.walk(ast.parse(source))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.mark.parametrize("module", sorted(SCOPE))
def test_every_scoped_function_exists(module):
    """A renamed function would leave the check below vacuous.  ``None``
    scopes the whole module."""
    src = (FR / module).read_text(encoding="utf-8")
    if SCOPE[module] is None:
        assert _defined(src)
        return
    missing = set(SCOPE[module]) - _defined(src)
    assert not missing, missing


@pytest.mark.parametrize("module", sorted(SCOPE))
def test_no_literal_reaches_the_screen(module):
    src = (FR / module).read_text(encoding="utf-8")
    scope = SCOPE[module]
    bad = offences(src, None if scope is None else set(scope))
    assert not bad, bad


def test_the_text_module_is_qt_free():
    tree = ast.parse((FR / "_step3_pr_text.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "Qt" not in (node.module or ""), node.module
            assert "qtpy" not in (node.module or ""), node.module
        if isinstance(node, ast.Import):
            assert not any("qt" in a.name.lower() for a in node.names)


def test_the_check_has_teeth():
    src = (
        "def f(self):\n"
        "    x.setText('Idle')\n"                                  # 1
        "    x.setToolTip(a if a else 'No description.')\n"        # 2
        "    self._append_log('n={n}'.format(n=1))\n"              # 3
        "    QLabel(f'Batch {n}')\n"                               # 4
        "    combo.addItem('Custom', 'custom')\n"                  # 5
        "    combo.addItems(['0: gray', '1: red'])\n"              # 6
        "    sb.setSuffix(' px')\n"                                # 7
        "    _make_subsection_header(self, 'Settings', guide_filename=g)\n"  # 8
        "    collapsible(T.A, 'pre_smoothing')\n"                  # 9
        "    x.setText(T.A)\n"
        "    combo.addItem(T.B, 'custom')\n"
        "    x.setText(T.A if c else T.B)\n"
        "    self._append_log(T.L.format(preset=p))\n"
        "    QLabel(T.C)\n"
        "    QMessageBox.warning(self, 'Busy', T.A)\n"            # 10
        "    QMessageBox.question(self, T.A, 'Sure?')\n"          # 11
        "    combo.setItemText(0, 'Custom')\n"                    # 12
        "    box.setTitle('Settings')\n"                          # 13
        "    w.setStatusTip(f'{n} cells')\n"                      # 14
        "    w.setPlaceholderText('Search')\n"                    # 15
        "    w.setWindowTitle('Set spacing')\n"                   # 16
        "    QMessageBox.information(self, T.A, T.B)\n"
        "    combo.setItemText(0, T.C)\n"
        "    self._pr_status('Ready')\n"                          # 17
        "    sub_header(self, 'Model', g, accent=a)\n"            # 18
        "    self._pr_status(T.A, 'warn')\n"
        "    sub_header(self, T.M, g, accent=a)\n")
    assert len(offences(src, None)) == 18

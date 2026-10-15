"""Help parity for ①, ②, ③ and the Export window.

The help lines and the Advanced labels are read from fixtures kept beside
this test (``fixtures/step4_help_lines.json`` and
``fixtures/step4_advanced_labels.json``: the agreed wording) and must
equal ``STEP4_HELP_LINES`` / ``STEP4_ADVANCED_LABELS``; every
visible ① / ② control carries its line as its tooltip; the painted parts of
a row (tick, chip, eye) answer through ``StackRowDelegate``.  The two hidden
knobs (the owner's ruling) have no line and are never shown.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from qtpy.QtCore import QPoint, QRect, Qt

from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"

PAINTED = {"row tick": "row_tick", "row chip": "row_chip",
           "row eye": "row_eye", "③ row click": "mt_row_click",
           "export row tick": "export_row_tick"}
#: the ① / ② widgets B1a builds (③ and the Export window are B1b's)
B1A_WIDGETS = (
    "run_picker_combo", "new_run_btn", "manage_runs_btn", "model_change_btn",
    "start_from_combo", "chan1_combo", "chan2_combo", "cell_width_um_sb",
    "normalize_cb", "rescale_cb", "predict_btn", "reconstruct_btn",
    "cancel_btn", "view_btn", "browse_workdir_btn", "add_folder_btn",
    "set_spacing_btn",
)
#: ③ and the Export window (B1b)
B1B_WIDGETS = (
    "extract_features_btn", "open_analysis_btn", "feature_reference_btn",
    "delete_bad_btn", "undo_btn", "smooth_strength_combo", "smooth_all_btn",
    "fix_by_hand_btn", "smooth_sigma_z_sb", "smooth_sigma_xy_sb",
    "smooth_preserve_topology_cb", "reset_original_btn", "save_tiers_btn",
    "finalize_btn", "export_btn", "export_features_btn", "export_meshes_btn",
)


def _norm(s: str) -> str:
    return " ".join(s.replace("\\|", "|").split())


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["data"]


def spec_help_lines() -> dict:
    """The agreed help line of every widget (constants already resolved)."""
    return _fixture("step4_help_lines.json")


def spec_advanced_labels() -> dict:
    """The agreed (label, engine) of every visible Advanced widget."""
    return {k: tuple(v) for k, v in
            _fixture("step4_advanced_labels.json").items()}


@pytest.mark.parametrize("name", ["step4_help_lines.json",
                                  "step4_advanced_labels.json"])
def test_the_fixture_exists(name):
    path = FIXTURES / name
    assert path.is_file() and path.stat().st_size > 0


#: Lines the code has changed ahead of the fixture (named, with the reason);
#: remove an entry once the fixture carries the same words.
PENDING_SPEC_EDITS = {
    # (empty: the fixture carries every line)
}


def test_the_spec_and_the_table_agree():
    spec = spec_help_lines()
    assert len(spec) > 60
    assert set(spec) == set(T.STEP4_HELP_LINES), (
        set(spec) ^ set(T.STEP4_HELP_LINES))
    bad = [k for k in spec if k not in PENDING_SPEC_EDITS
           and _norm(T.STEP4_HELP_LINES[k]) != spec[k]]
    assert not bad, bad
    for k, text in PENDING_SPEC_EDITS.items():
        assert T.STEP4_HELP_LINES[k] == text, k
        assert spec[k] != _norm(text), (
            f"the agreed wording now has {k}'s line: drop it from "
            f"PENDING_SPEC_EDITS")


def test_the_advanced_labels_are_the_specs():
    spec = spec_advanced_labels()
    assert len(spec) == 43
    assert spec == T.STEP4_ADVANCED_LABELS


def test_the_hidden_knobs_have_no_line():
    for w in T.STEP4_HIDDEN_KNOBS:
        assert w not in T.STEP4_HELP_LINES


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif",))
    w = panel(qapp, ws)
    yield w
    close(w)


def test_every_visible_control_carries_its_line(w):
    for name in B1A_WIDGETS + B1B_WIDGETS + tuple(T.STEP4_ADVANCED_LABELS):
        widget = getattr(w, name)
        assert widget.toolTip() == T.STEP4_HELP_LINES[name], name


def test_every_advanced_control_shows_its_plain_label(w):
    for name, (plain, engine) in T.STEP4_ADVANCED_LABELS.items():
        lbl = w._adv_engine_labels[name]
        text = lbl.text().replace("​", "")
        assert plain in text and engine in text, name


def test_the_hidden_knobs_are_hidden(w):
    for name in T.STEP4_HIDDEN_KNOBS:
        assert not getattr(w, name).isVisibleTo(w)


def test_the_painted_parts_answer(qapp, w):
    """tick / eye / chip of ② and ① rows, and ③'s whole row."""
    from qtpy.QtWidgets import QStyleOptionViewItem

    from napari_epicycle.segmentation import _step4_models as M
    c = w.controller
    snap = M.model_snapshot("cyto3", "cyto3", kind="pretrained")
    c.state_manager.set_stack_status("P1.tif", "reconstructed",
                                     made_with=snap, labels_path="x")
    c._refresh_stacks()
    pump(qapp)
    for lw, delegate in ((w.stacks_list, w._seg_delegate),
                         (w.session_stacks_list, w._session_delegate)):
        idx = lw.model().index(0, 0)
        opt = QStyleOptionViewItem()
        opt.rect = lw.visualRect(idx)
        opt.font = lw.font()
        opt.widget = lw
        g = delegate.layout(opt.rect, idx.data(Qt.UserRole + 2), opt.font)
        assert delegate.help_key_at(opt, idx, g["tick"].center()) == \
            "row_tick"
        assert delegate.help_key_at(opt, idx, g["eye"].center()) == "row_eye"
        if g["chips"]:
            assert delegate.help_key_at(opt, idx,
                                        g["chips"][0][0].center()) == \
                "row_chip"
    lw = w.qc_stacks_list
    idx = lw.model().index(0, 0)
    opt = QStyleOptionViewItem()
    opt.rect = lw.visualRect(idx)
    opt.widget = lw
    assert w._qc_delegate.help_key_at(opt, idx, QPoint(5, 5)) == \
        "mt_row_click"


def test_every_line_has_a_home(w):
    """Every help-line key is a built widget or a painted part (nothing is a
    line for a control that does not exist)."""
    painted = set(PAINTED.values())
    for key in T.STEP4_HELP_LINES:
        assert key in painted or hasattr(w, key), key


def test_the_export_row_tick_answers(qapp, w):
    from qtpy.QtCore import QEvent, QPoint
    from qtpy.QtGui import QHelpEvent
    from qtpy.QtWidgets import QStyleOptionViewItem

    dlg = w.export_dialog
    dlg.set_rows([{"name": "P1.tif", "word": "finalized", "dots": "",
                   "colour": "#80deea", "icon": "mdi6.diamond-stone",
                   "can_go": True, "why": ""}])
    dlg.show()
    pump(qapp)
    lw = w.export_rows_list
    idx = lw.model().index(0, 0)
    opt = QStyleOptionViewItem()
    opt.rect = lw.visualRect(idx)
    shown = []
    import napari_epicycle.segmentation._export_dialog as X
    orig = X._show_tip
    try:
        X._show_tip = lambda pos, text, view: shown.append(text)
        ev = QHelpEvent(QEvent.ToolTip, opt.rect.center(),
                        lw.viewport().mapToGlobal(opt.rect.center()))
        lw.itemDelegate().helpEvent(ev, lw, opt, idx)
    finally:
        X._show_tip = orig
        dlg.hide()
    assert shown == [T.STEP4_HELP_LINES["export_row_tick"]]


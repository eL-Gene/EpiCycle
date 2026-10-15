"""③ Measure & tidy and the Export window, built (brick B1b).

The real panel with its controller.  ③ is the mockup's two pinned columns:
LEFT the reconstructed-only list filling the column with the stack card at
its foot and the "not reconstructed yet" line; RIGHT four blocks (a header
row with a chip, one desc line, one control row each) and a closed
Advanced in a scroll area, over a pinned foot.

Heights are measured at the width the user gets (height-for-width at 322 px
for the right column, rows at 218 px), in state D (idle) and D3 (Measure
running), with every optional line on, and asserted against the mockup's
ceilings (``MOCKUP_CEILINGS``, never from this build).  As in B1a's
skeleton test, every font-dependent number is measured in a SUBPROCESS
pointed at the system fonts with napari's Windows face.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump
from .test_step4_mockup_ceilings import MOCKUP_CEILINGS as CEIL

SECTION_W = 572
_FONT_DIR = "C:/Windows/Fonts"
_REPO = Path(__file__).resolve().parents[2]

#: the mockup's D / D3 names (lengths matter for the wrapped lines)
NAMES = ("WholeScale_patch_1.tiff", "WholeScale_patch_2.tiff",
         "WholeScale_patch_3.tiff", "Wound_patch_2.tiff",
         "Wound_patch_3.tiff", "Wound_patch_1.tiff")
LOADED = "WholeScale_patch_2.tiff"


def _hfw(widget, width):
    if widget.hasHeightForWidth():
        return widget.heightForWidth(width)
    return widget.sizeHint().height()


def _open(w, qapp, index):
    w.accordion.setCurrentIndex(index)
    pump(qapp, 10)


def _big_volume():
    """A 212×1024×1024 volume that costs no memory (only its shape is
    read by the stack card)."""
    return np.lib.stride_tricks.as_strided(
        np.zeros(1, np.uint32), shape=(212, 1024, 1024), strides=(0, 0, 0))


def build(qapp, tmp_path):
    """State D of the v7 mockup: five reconstructed stacks (three
    finalized, one measured and loaded with unsaved edits, one
    reconstructed), one stack not reconstructed yet."""
    tmp_path = Path(tmp_path)
    ws = make_workspace(tmp_path, patches=NAMES, voxel=(1.0, 0.33, 0.33))
    w = panel(qapp, ws)
    w.resize(SECTION_W, 842)
    c = w.controller
    status = {NAMES[0]: "finalized", NAMES[1]: "qc_extracted",
              NAMES[2]: "finalized", NAMES[3]: "reconstructed",
              NAMES[4]: "finalized"}
    for name, st in status.items():
        run = ws / "artifacts" / "step4" / "runs" / Path(name).stem
        run.mkdir(parents=True, exist_ok=True)
        lab = run / "labels_3d.tif"
        tifffile.imwrite(str(lab), np.zeros((2, 4, 4), np.uint32))
        c.state_manager.set_stack_status(
            name, st, labels_path=str(lab), n_cells=4616,
            stack_shape=[2, 212, 1024, 1024])
    c.state_manager.update_record(
        LOADED, measured_at="2026-09-28T12:02:00+00:00",
        measured_cells=4653,
        measured_spacing={"zyx": [1.0, 0.33, 0.33], "source": "step1"},
        original_labels_path=str(ws / "orig.tif"))
    (ws / "orig.tif").write_bytes(b"x")
    c._refresh_stacks()
    c._apply_loaded(LOADED, {"labels": np.zeros((2, 4, 4), np.uint32),
                             "n_cells": 4616, "backup": "", "raw": None,
                             "source_path": ""})
    c._labels_volume = _big_volume()
    c._n_cells = 4616
    tiers = ["good"] * 1204 + [""] * 3412
    c._qc_features_df = pd.DataFrame({"label": np.arange(1, 4617),
                                      "qc_tier": tiers,
                                      "label_status": ["active"] * 4616})
    c._labels_dirty = True
    c._pending_edits = {"labels_changed_at": "2026-09-28T12:12:00+00:00",
                        "smoothed_at": "2026-09-28T12:12:00+00:00",
                        "n_thin_kept": 3, "_stack": LOADED}
    w.show_mt_status(T.say(T.MT_SMOOTHED, n=4616, v=18402))
    w.show_qc_bar(value=100, text=T.QC_BAR_IDLE)
    c._refresh_mt_panel()
    _open(w, qapp, 2)
    return w


def _busy(w, on: bool, longest: bool = False):
    """Stage D3: Measure running (the flag, the lock line, the bar)."""
    c = w.controller
    c._set_step4_flag("_measure_running", bool(on))     # the real push
    if on and longest:
        # R1-9: measured with the LONGEST lock sentence in the lock line
        from .test_step4_panel_skeleton import LONGEST_LOCK
        w.set_lock_lines(LONGEST_LOCK)
    if on:
        w.show_qc_bar(value=62, text=T.BUSY_MEASURING.format(pct=62))


@pytest.fixture
def w(qapp, tmp_path):
    w = build(qapp, tmp_path)
    yield w
    close(w)


# --------------------------------------------------------------- the build


def test_two_pinned_columns_like_segment(qapp, w):
    sp = w.mt_splitter
    assert sp.orientation() == Qt.Horizontal
    assert not sp.childrenCollapsible()
    assert sp.handleWidth() == 6
    assert sp.sizes() == [224, 330]
    assert w.mt_left.layout().contentsMargins().right() == 6
    assert w.mt_right.layout().contentsMargins().left() == 8
    assert w.mt_left.minimumWidth() == 200
    assert w.mt_right.minimumWidth() == 300
    assert w.mt_foot.parent() is w.mt_right
    assert not w.mt_right_content.isAncestorOf(w.save_tiers_btn)
    assert w.mt_right_scroll.horizontalScrollBarPolicy() == \
        Qt.ScrollBarAlwaysOff


def test_the_stack_card_is_at_the_foot_of_the_left_column(qapp, w):
    lay = w.mt_left.layout()
    order = [lay.itemAt(i).widget() for i in range(lay.count())]
    assert order.index(w.qc_stacks_list) < order.index(w.stack_card) < \
        order.index(w.mt_not_recon_label)
    assert lay.stretch(order.index(w.qc_stacks_list)) == 1  # fills
    assert w.stack_card_name.text() == LOADED
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_UNSAVED
    assert "4,616 cells" in w.stack_card_line.text()
    assert w.mt_not_recon_label.text() == T.MT_NOT_RECONSTRUCTED.format(
        stack=NAMES[5])


def test_the_list_is_reconstructed_only_with_no_ticks(qapp, w):
    lw = w.qc_stacks_list
    names = [lw.item(i).data(Qt.UserRole) for i in range(lw.count())]
    assert NAMES[5] not in names and len(names) == 5
    for i in range(lw.count()):
        assert not lw.item(i).flags() & Qt.ItemIsUserCheckable
    loaded = [lw.item(i) for i in range(lw.count())
              if lw.item(i).data(Qt.UserRole) == LOADED][0]
    assert loaded.isSelected()
    assert T.ROW_UNSAVED in loaded.data(Qt.UserRole + 2)["warns_narrow"]


def test_four_blocks_their_chips_and_the_foot(qapp, w):
    assert w.mt_chip_measure.text().startswith("stale · ")
    assert w.mt_chip_measure.text().endswith("4,653 cells")
    html = w.mt_chip_review.text()
    for part in ("1,204", "good", "0", "bad", "3,412", "unmarked"):
        assert part in html
    assert w.mt_chip_tidy.text().endswith(" · 3 thin kept")
    assert w.mt_chip_fix.text() == T.MT_CHIP_FIX_LATER
    assert w.extract_features_btn.text() == T.MEASURE_BTN_AGAIN
    assert not w.fix_by_hand_btn.isEnabled()
    assert w.finalize_note.text() == T.FINALIZE_NEEDS
    assert not w.finalize_btn.isEnabled() and w.save_tiers_btn.isEnabled()
    assert w.export_btn.isEnabled()
    assert w.qc_progress.format() == T.QC_BAR_IDLE
    assert w.export_btn.height() <= 22
    assert not w._mt_adv_collapsible.isExpanded()


def test_d3_while_measure_runs(qapp, w):
    _busy(w, True)
    pump(qapp)
    assert w.mt_lock_line.text() == T.STEP4_LOCK_MEASURE
    assert w.session_lock_line.text() == T.STEP4_LOCK_MEASURE
    assert not w.qc_stacks_list.isEnabled()
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_MEASURING
    assert w.extract_features_btn.text() == T.MEASURE_BTN_RUNNING
    assert not w.mt_chip_measure.text().startswith("stale")
    assert w.finalize_note.isHidden()
    for name in ("extract_features_btn", "open_analysis_btn",
                 "delete_bad_btn", "undo_btn", "smooth_all_btn",
                 "smooth_strength_combo", "save_tiers_btn", "finalize_btn",
                 "export_btn", "reset_original_btn"):
        assert not getattr(w, name).isEnabled(), name
    assert w.feature_reference_btn.isEnabled()      # the one live control
    busy_text = T.BUSY_MEASURING.format(pct=62)
    assert w.qc_progress.format() == busy_text
    _busy(w, False)
    assert w.mt_lock_line.text() == ""


NEW_NAMES = (
    "stack_card", "stack_card_name", "stack_card_pill", "stack_card_line",
    "undo_btn", "smooth_strength_combo", "fix_by_hand_btn",
    "mt_chip_measure", "mt_chip_review", "mt_chip_tidy", "mt_chip_fix",
    "finalize_note", "qc_progress", "export_btn", "qc_status_line",
    "export_dialog", "export_features_btn", "export_status_label",
    "export_rows_list", "mt_lock_line",
)
KEPT_NAMES = (
    "qc_stacks_list", "extract_features_btn", "open_analysis_btn",
    "feature_reference_btn", "delete_bad_btn", "smooth_sigma_z_sb",
    "smooth_sigma_xy_sb", "smooth_preserve_topology_cb", "smooth_all_btn",
    "save_tiers_btn", "finalize_btn", "export_meshes_btn",
    "reset_original_btn", "_channel_container", "qc_axes_label",
)
GONE_NAMES = (
    "load_reconstruction_btn", "qc_status_label", "qc_feature_status",
    "smooth_status_label", "spacing_z_sb", "spacing_y_sb", "spacing_x_sb",
    "spacing_auto_btn", "z_orientation_combo", "extract_all_features_btn",
    "extract_progress", "smooth_progress", "mesh_progress", "export_all_btn",
    "scope_per_stack_rb", "scope_population_rb", "_qc_extract", "_qc_triage",
    "_qc_smooth", "_qc_export",
)


def test_the_widget_names(w):
    for name in NEW_NAMES + KEPT_NAMES:
        assert hasattr(w, name), name
    for name in GONE_NAMES:
        assert not hasattr(w, name), name
    # the Export window's widgets live in the window
    assert w.export_meshes_btn.window() is w.export_dialog
    assert w.export_features_btn.window() is w.export_dialog
    assert not w.export_dialog.isVisible()


def test_the_topology_tick_says_one_piece(w):
    labels = [c.text() for c in w.mt_right_content.findChildren(type(
        w.qc_axes_label)) if c.text() == T.TIDY_TOPOLOGY_LABEL]
    assert labels


def test_strength_sets_the_sigmas_and_a_hand_edit_reads_custom(qapp, w):
    c = w.smooth_strength_combo
    assert c.currentText() == T.SMOOTH_STRENGTHS[1]
    # µm: the loaded stack has 1.0 × 0.33 µm per voxel (Step 1)
    assert w.smooth_sigma_z_sb.suffix() == T.SIGMA_SUFFIX_UM
    assert w.smooth_sigmas_voxels() == (3.0, 1.5)
    assert abs(w.smooth_sigma_z_sb.value() - 3.0) < 1e-6
    assert abs(w.smooth_sigma_xy_sb.value() - 1.5 * 0.33) < 0.01
    w._on_strength_chosen(2)
    assert w.smooth_sigmas_voxels() == (6.0, 3.0)
    w._on_strength_chosen(0)
    assert w.smooth_sigmas_voxels() == (1.5, 0.75)
    w.smooth_sigma_xy_sb.setValue(w.smooth_sigma_xy_sb.value() + 0.5)
    assert c.currentText() == T.SMOOTH_STRENGTH_CUSTOM
    w._on_strength_chosen(1)
    assert c.findText(T.SMOOTH_STRENGTH_CUSTOM) < 0
    w.set_sigma_spacing(None)                       # a stack in voxels
    assert w.smooth_sigma_z_sb.suffix() == T.SIGMA_SUFFIX_VOX
    assert w.smooth_sigma_z_sb.value() == 3.0


def test_the_export_window_is_480_and_built_once(qapp, w):
    dlg = w.export_dialog
    assert dlg.width() == CEIL["STEP4_EXPORT_W"] == 480
    assert dlg.minimumSizeHint().width() <= 480
    assert not dlg.isModal()
    w.controller._open_export_window()
    pump(qapp)
    assert dlg.isVisible() and w.export_dialog is dlg
    dlg.close_btn.click()
    assert not dlg.isVisible()


# ------------------------------------------------ the measured ceilings


def measure(w, qapp):
    out = {}
    _open(w, qapp, 2)
    col = CEIL["STEP4_SEG_RIGHT_COL_W"]
    out["right_min_w"] = w.mt_right_content.minimumSizeHint().width()
    out["foot_min_w"] = w.mt_foot.minimumSizeHint().width()
    content = w.accordion._sections[2]._scroll.widget()
    out["section_min_w"] = content.minimumSizeHint().width()
    out["left_w"] = w.mt_left.width() - 6
    out["left_list_min_w"] = w.qc_stacks_list.minimumSizeHint().width()
    out["right"] = _hfw(w.mt_right_content, col)
    out["foot"] = _hfw(w.mt_foot, col)
    lw = w.qc_stacks_list
    visible, used = 0, 0
    for i in range(lw.count()):
        h = lw.sizeHintForRow(i)
        if used + h > lw.viewport().height():
            break
        used += h
        visible += 1
    out["rows_visible"] = visible
    out["row_hs"] = [lw.sizeHintForRow(i) for i in range(lw.count())]
    out["list_vp_h"] = lw.viewport().height()
    out["vscroll_policy"] = int(lw.verticalScrollBarPolicy().value
                                if hasattr(lw.verticalScrollBarPolicy(),
                                           "value")
                                else lw.verticalScrollBarPolicy())
    out["vscroll_max"] = lw.verticalScrollBar().maximum()
    out["n_rows"] = lw.count()
    out["lock"] = _hfw(w.mt_lock_line, 560) if not w.mt_lock_line.isHidden()         else 0
    w.set_channel_rows(2, axes_label="ZCYX")
    w._mt_adv_collapsible.expand(animate=False)
    pump(qapp, 5)
    out["right_adv"] = _hfw(w.mt_right_content, col)
    out["right_adv_min_w"] = w.mt_right_content.minimumSizeHint().width()
    w._mt_adv_collapsible.collapse(animate=False)
    pump(qapp, 5)
    dlg = w.export_dialog
    w.controller._refresh_export_window()
    pump(qapp, 5)
    out["export_w"] = dlg.width()
    out["export_min_w"] = dlg.minimumSizeHint().width()
    return out


def measure_all(qapp):
    out = {"font": qapp.font().family()}
    with tempfile.TemporaryDirectory() as tmp:
        w = build(qapp, tmp)
        try:
            for state in ("idle", "busy"):
                _busy(w, state == "busy", longest=True)
                pump(qapp, 10)
                out[state] = measure(w, qapp)
            _busy(w, False)
        finally:
            close(w)
            pump(qapp, 5)
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
from tests.segmentation import test_step4_mt_skeleton as t
print("S4JSON" + json.dumps(t.measure_all(app)), flush=True)
os._exit(0)
"""


@pytest.fixture(scope="module")
def ceilings():
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
    got = json.loads(lines[-1][len("S4JSON"):])
    print(f"[CEIL] ③ {json.dumps(got)}")
    return got


@pytest.fixture(params=["idle", "busy"])
def state(request):
    return request.param


def test_measured_in_the_users_font(ceilings):
    if sys.platform == "win32" and os.path.isdir(_FONT_DIR):
        assert ceilings["font"] == "Segoe UI"


def test_nothing_scrolls_sideways(ceilings, state):
    got = ceilings[state]
    assert got["right_min_w"] <= CEIL["STEP4_SEG_RIGHT_COL_W"]
    assert got["foot_min_w"] <= CEIL["STEP4_SEG_RIGHT_COL_W"]
    assert got["right_adv_min_w"] <= CEIL["STEP4_SEG_RIGHT_COL_W"]
    assert got["section_min_w"] <= SECTION_W


def test_the_left_rows_lay_out_at_218(ceilings, state):
    got = ceilings[state]
    assert got["left_w"] == CEIL["STEP4_LEFT_COL_W"]
    assert got["left_list_min_w"] <= CEIL["STEP4_LEFT_COL_W"]


def test_right_column_state_d_and_d3(ceilings, state):
    """D idle against STEP4_MT_RIGHT_D_PX, D3 (Measure running) against
    STEP4_MT_RIGHT_D3_PX; the maximum of the two against D's."""
    h = ceilings[state]["right"]
    print(f"[CEIL] ③ right {state}: {h}")
    key = "STEP4_MT_RIGHT_D_PX" if state == "idle" else \
        "STEP4_MT_RIGHT_D3_PX"
    assert h <= CEIL[key]
    assert max(ceilings["idle"]["right"], ceilings["busy"]["right"]) <= \
        CEIL["STEP4_MT_RIGHT_D_PX"]


def test_the_pinned_foot(ceilings, state):
    h = ceilings[state]["foot"]
    print(f"[CEIL] ③ foot {state}: {h}")
    assert h <= CEIL["STEP4_MT_FOOT_PX"]


#: Decided 2026-10-10 (A3): while a lock line shows, ③ keeps TWO
#: visible rows (idle keeps three, ``STEP4_MT_LIST_MIN_ROWS``); the narrow
#: rows grew because the voxel-size line wraps in the 74 px text column
#: (rows of 133 px, 394 px of list busy).  The list scrolls.
STEP4_MT_LIST_MIN_ROWS_BUSY = 2


def test_three_rows_show_at_the_default_height(ceilings, state):
    print(f"[CEIL] mt rows {state}: {ceilings[state]['row_hs']} in "
          f"{ceilings[state]['list_vp_h']} px")
    need = (CEIL["STEP4_MT_LIST_MIN_ROWS"] if state == "idle"
            else STEP4_MT_LIST_MIN_ROWS_BUSY)
    assert ceilings[state]["rows_visible"] >= need
    # the rows that do not fit are reached by scrolling the list
    from qtpy.QtCore import Qt
    assert ceilings[state]["vscroll_policy"] != int(
        Qt.ScrollBarAlwaysOff.value)
    if ceilings[state]["n_rows"] >= 5:
        assert ceilings[state]["vscroll_max"] > 0, state


#: ③ with Advanced open has no mockup state, so no mockup number
#: (``STEP4_MT_RIGHT_D_ADV_PX`` is None in the data file).  Owner decision
#: 60 (2026-10-05): the cap is the measured build value, 681 px (322 px,
#: two channel rows, idle and busy) -- it was 715 (681 + 5 %) to approve.
STEP4_MT_RIGHT_D_ADV_PX = 681


def test_advanced_open_holds_the_build_ceiling(ceilings, state):
    assert CEIL["STEP4_MT_RIGHT_D_ADV_PX"] is None
    h = ceilings[state]["right_adv"]
    print(f"[CEIL] ③ right, Advanced open, {state}: {h} "
          f"(ceiling {STEP4_MT_RIGHT_D_ADV_PX})")
    assert ceilings[state]["right"] < h <= STEP4_MT_RIGHT_D_ADV_PX


def test_the_export_window_width(ceilings, state):
    got = ceilings[state]
    assert got["export_w"] == CEIL["STEP4_EXPORT_W"]
    assert got["export_min_w"] <= CEIL["STEP4_EXPORT_W"]


def test_the_lock_line_in_measure_and_tidy(ceilings):
    """R1-9: ③'s lock line with the longest sentence holds the lock-line
    ceiling (560 px, at most 3 lines)."""
    h = ceilings["busy"]["lock"]
    print(f"[CEIL] ③ lock line busy: {h}")
    assert 0 < h <= CEIL["STEP4_LOCK_LINE_PX"]

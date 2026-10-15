"""Every grey ③ / Export control and where its reason is ON SCREEN
(Qt shows no tooltip on a disabled widget).  The real panel with its
controller, one grey control per test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))
    w = panel(qapp, ws)
    c = w.controller
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    lab = np.zeros((4, 8, 8), np.uint32)
    lab[1, 1:3, 1:3] = 1
    lab[2, 4:6, 4:6] = 2
    tifffile.imwrite(str(run / "labels_3d.tif"), lab)
    orig = run / "labels_3d_original.tif"
    tifffile.imwrite(str(orig), lab)
    c.state_manager.set_stack_status(
        "P1.tif", "reconstructed", labels_path=str(run / "labels_3d.tif"),
        original_labels_path=str(orig))
    c._refresh_stacks()
    pump(qapp)
    w._lab = lab
    yield w
    close(w)


def _load(w, qapp, measured=True, tiers=("good", "")):
    c = w.controller
    c._apply_loaded("P1.tif", {"labels": w._lab.copy(), "n_cells": 2,
                               "backup": "", "raw": None, "source_path": ""})
    if measured:
        c._qc_features_df = pd.DataFrame({"label": [1, 2],
                                          "qc_tier": list(tiers),
                                          "label_status": ["active"] * 2})
        c.state_manager.update_record(
            "P1.tif", measured_at="2026-09-28T12:00:00+00:00",
            measured_cells=2, measured_spacing=c.state_manager.get_record(
                "P1.tif").get("spacing"))
    c._refresh_mt_panel()
    pump(qapp)
    return c


def test_nothing_loaded(qapp, w):
    """Measure / Smooth / strength: the card says "No stack loaded.";
    Save: the card says "Load a stack to save it."."""
    for name in ("extract_features_btn", "smooth_all_btn",
                 "smooth_strength_combo", "save_tiers_btn", "finalize_btn",
                 "open_analysis_btn", "delete_bad_btn",
                 "reset_original_btn"):
        assert not getattr(w, name).isEnabled(), name
    assert w.stack_card_name.text() == T.MT_NOTHING_LOADED
    assert T.SAVE_NO_STACK in w.stack_card_line.text()
    assert not w.stack_card_name.isHidden()
    assert not w.stack_card_line.isHidden()


def test_export_is_live_with_nothing_loaded(qapp, w):
    assert w.export_btn.isEnabled()
    assert w.undo_btn.isEnabled()              # never grey outside the lock


def test_not_measured(qapp, w):
    """Open Workbench: the Measure chip says "not measured"; Finalize:
    the amber foot line says "Measure this stack first."."""
    _load(w, qapp, measured=False)
    assert not w.open_analysis_btn.isEnabled()
    assert w.mt_chip_measure.text() == T.MT_CHIP_NEVER
    assert not w.finalize_btn.isEnabled()
    assert w.finalize_note.text() == T.FINALIZE_NEEDS_MEASURE
    assert not w.finalize_note.isHidden()
    assert w.extract_features_btn.isEnabled()
    assert w.extract_features_btn.text() == T.MEASURE_BTN_FIRST


def test_no_cell_marked_bad(qapp, w):
    _load(w, qapp, tiers=("good", ""))
    assert not w.delete_bad_btn.isEnabled()
    assert "0</span> bad" in w.mt_chip_review.text()
    _load(w, qapp, tiers=("bad", ""))
    assert w.delete_bad_btn.isEnabled()


def test_nothing_to_save(qapp, w):
    _load(w, qapp)
    assert not w.save_tiers_btn.isEnabled()
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_SAVED
    w.controller._tiers_dirty = True                   # a mark
    w.controller._refresh_mt_panel()
    assert w.save_tiers_btn.isEnabled()
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_UNSAVED_MARKS
    assert w.finalize_btn.isEnabled()                  # I-7: marks never grey


def test_stale_measurements(qapp, w):
    c = _load(w, qapp)
    assert w.finalize_btn.isEnabled() and w.finalize_note.isHidden()
    c._labels_dirty = True                              # an edit
    c._refresh_mt_panel()
    assert not w.finalize_btn.isEnabled()
    assert w.finalize_note.text() == T.FINALIZE_NEEDS
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_UNSAVED
    assert w.mt_chip_measure.text().startswith("stale · ")


def test_a_voxel_size_change_makes_it_stale(qapp, w):
    c = _load(w, qapp)
    c.state_manager.update_record(
        "P1.tif", spacing={"zyx": [3.0, 0.5, 0.5], "source": "user"})
    c._refresh_mt_panel()
    assert w.finalize_note.text() == T.FINALIZE_NEEDS
    assert w.stack_card_note.text() == T.MT_SPACING_CHANGED
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_STALE


def test_fix_by_hand_is_always_grey_and_says_why(qapp, w):
    _load(w, qapp)
    assert not w.fix_by_hand_btn.isEnabled()
    assert w.mt_chip_fix.text() == T.MT_CHIP_FIX_LATER
    labels = [lbl.text() for lbl in w.mt_right_content.findChildren(
        type(w.qc_axes_label))]
    assert T.FIX_DESC in labels
    # no handler is connected
    assert w.fix_by_hand_btn.receivers(w.fix_by_hand_btn.clicked) == 0


def test_reset_without_a_backup(qapp, w):
    c = w.controller
    orig = Path(c.state_manager.get_record("P1.tif")[
        "original_labels_path"])
    orig.unlink()
    _load(w, qapp)
    assert not w.reset_original_btn.isEnabled()
    assert w.qc_status_line.text() == T.RESET_NO_BACKUP.format(
        stack="P1.tif")


def test_undo_with_nothing_says_so(qapp, w):
    w.undo_btn.click()
    assert w.qc_status_line.text() == T.MT_UNDO_EMPTY


def test_the_export_window_reasons(qapp, w):
    c = w.controller
    c._open_export_window()
    pump(qapp)
    dlg = w.export_dialog
    assert not w.export_features_btn.isEnabled()
    assert w.export_status_label.text() == T.EXPORT_BLOCK
    assert not w.export_meshes_btn.isEnabled()
    assert dlg.mesh_target_label.text() == T.EXPORT_MESH_NONE
    _load(w, qapp)
    assert w.export_meshes_btn.isEnabled()
    assert dlg.mesh_target_label.text() == T.EXPORT_MESH_TARGET.format(
        stack="P1.tif")
    dlg.set_busy(True)
    assert w.export_status_label.text() == T.EXPORT_RUNNING
    assert not w.export_meshes_btn.isEnabled()
    assert dlg.close_btn.isEnabled()
    dlg.set_busy(False)
    dlg.hide()


def test_every_grey_reason_is_a_constant():
    """The sentences above exist (a rename would hide a reason)."""
    for name in ("MT_NOTHING_LOADED", "SAVE_NO_STACK", "MT_CHIP_NEVER",
                 "MT_CHIP_REVIEW", "MT_UNDO_EMPTY", "MT_STACK_PILL_SAVED",
                 "FINALIZE_NEEDS_MEASURE", "FINALIZE_NEEDS",
                 "MT_CHIP_FIX_LATER", "FIX_DESC", "RESET_NO_BACKUP",
                 "EXPORT_BLOCK", "EXPORT_NONE_TICKED", "EXPORT_RUNNING",
                 "EXPORT_MESH_NONE"):
        assert isinstance(getattr(T, name), str) and getattr(T, name)

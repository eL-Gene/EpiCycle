"""The B2 fix pass's UX findings on the Add page (U-H2, U-M1..U-M4).

* U-H2 a refused run's reason stays in the Markers help slot -- it used to
  be written there and overwritten by the next refresh, so S1 said "the
  reason is under Markers" over the ordinary help;
* U-M1 a preset switch keeps a marker source the user chose;
* U-M2 the counts note is elided to two lines, top-aligned, the full text
  in its tooltip (it was only height-capped, and vertically centred, so it
  was cut at the top AND the bottom);
* U-M3 REVIEW's S1 follows the list selection, with a sentence for "more
  than one selected";
* U-M4 a guide worker's raw error never reaches the page, and a refusal
  carries a file's basename, never its full path.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("qtpy")

from packages.epicycle_annotator import add_preview as AP  # noqa: E402

from .add_harness import Harness, blank, install_fakes  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    w.resize(340, 350)
    w.show()
    qapp.processEvents()
    w.focus_tab("Add")
    qapp.processEvents()
    w.refresh_add_image_combo(["nuc", "mem"])
    w.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_RUN.format(name="r1"), True, ("labels", "r1")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None)),
    ], box_active=True, voxel_size_known=True)
    yield w
    w.hide()
    w.setParent(None)


# --------------------------------------------------------------------------- #
# U-H2 -- the refused run's reason stays under Markers
# --------------------------------------------------------------------------- #
def test_a_refusal_survives_the_next_refresh(panel):
    panel.set_add_preset("cells")
    panel.add_marker_source_combo.setCurrentIndex(0)          # the run
    panel.set_add_source_note("That run's mask is a different size.")
    panel.update_add_ui_state(status_text=AP.ADD_S1_RUN_UNUSABLE)
    assert panel._add_source_help.text() == (
        "That run's mask is a different size.")
    assert not panel._add_source_help.isHidden()


def test_the_note_goes_with_a_source_change_or_an_empty_note(panel):
    panel.set_add_preset("cells")
    panel.add_marker_source_combo.setCurrentIndex(0)
    panel.set_add_source_note("refused")
    panel.add_marker_source_combo.setCurrentIndex(2)          # clicks
    panel.add_marker_source_combo.setCurrentIndex(0)          # the run again
    assert panel._add_source_help.text() != "refused"
    panel.set_add_source_note("refused")
    panel.set_add_source_note("")
    panel.update_add_ui_state(status_text="x")
    assert panel._add_source_help.text() != "refused"


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(blank())
    notes = []
    hh.adv.set_add_source_note = notes.append
    hh.notes = notes
    return hh


def test_the_orchestrator_hands_the_panel_a_basename_not_a_path(h):
    h._add_run_guide_token = 1
    path = r"C:\Users\someone\Desktop\workspace\artifacts\step3\runs\r1\labels.tif"
    h._on_add_guide_refused(
        "There is no file at {}. It may have been moved or renamed - pick "
        "the mask file again.".format(path))
    assert h._add_run_state == "refused"
    note = h.notes[-1]
    assert "labels.tif" in note
    assert "Desktop" not in note and "\\" not in note


def test_a_guide_error_is_a_fixed_sentence(h):
    h.ctx["spacing"] = (0.5, 0.5, 0.5)
    h.adv.preset = "cells"
    h.adv.source = ("labels", "r1")
    h.place_box((slice(0, 6), slice(0, 10), slice(0, 10)))
    h._on_add_guide_error("ValueError: boom\nTraceback (most recent call ...")
    assert h._add_run_state == "refused"
    assert h.notes[-1] == AP.ADD_RUN_READ_FAILED
    assert h.adv.last("status_text") == AP.ADD_S1_RUN_UNUSABLE


def test_a_run_reset_clears_the_note(h):
    h.notes.append("old")
    h._add_run_guide_reset()
    assert h.notes[-1] == ""


# --------------------------------------------------------------------------- #
# U-M1 -- a chosen marker source survives a preset round trip
# --------------------------------------------------------------------------- #
def test_a_chosen_source_survives_cells_nuclei_cells(panel):
    panel.set_add_preset("cells")
    assert panel.get_add_marker_source()[0] == "nuclei_masks"   # the default
    panel.add_marker_source_combo.setCurrentIndex(2)           # the user
    assert panel.get_add_marker_source()[0] == "paint"
    panel.add_preset_combo.setCurrentIndex(0)                  # Nuclei
    panel.add_preset_combo.setCurrentIndex(1)                  # Cells
    assert panel.get_add_marker_source()[0] == "paint"


def test_an_untouched_source_still_gets_the_default(panel):
    panel.set_add_preset("cells")
    panel.clear_add_touched()
    panel.add_preset_combo.setCurrentIndex(0)
    panel.add_preset_combo.setCurrentIndex(1)
    assert panel.get_add_marker_source()[0] == "nuclei_masks"


# --------------------------------------------------------------------------- #
# U-M2 -- the counts note, two lines, top-aligned, full text in the tooltip
# --------------------------------------------------------------------------- #
def test_the_counts_note_is_elided_to_two_lines(panel, qapp):
    """Now ONE line: take-off step (3), taken in the live-shot
    fix pass (V9) when the 312 px viewport put 7c over the ceiling -- the
    cap is ADD_NOTES_LINES, the rest of the contract unchanged."""
    from qtpy.QtCore import Qt

    counts = {k: 12 for k, _m, _o in AP._COUNT_PARTS}
    full = AP.counts_note(counts)
    panel.update_add_ui_state(candidate_ids=[1, 2], candidate_sizes={1: 5, 2: 6},
                              notes=[("counts", full, False)])
    qapp.processEvents()
    lb = panel.add_notes_label
    shown = lb.text()
    lines = shown.split("\n")
    assert AP.ADD_NOTES_LINES == 1
    assert len(lines) <= AP.ADD_NOTES_LINES
    assert shown.endswith("…")
    assert lb.toolTip() == full
    fm = lb.fontMetrics()
    width = lb.contentsRect().width()
    assert all(fm.horizontalAdvance(line) <= width for line in lines)
    assert lb.alignment() & Qt.AlignTop
    assert lb.height() >= AP.ADD_NOTES_LINES * fm.lineSpacing()


def test_a_short_note_is_not_elided(panel, qapp):
    """One count part, the shortest real one: since the note is ONE line
    (live-shot V9) the 46-character dark-space part sits within 2 px of the
    offscreen width and elides depending on the page's scrollbar."""
    short = AP.counts_note({"n_below_min_size": 2})
    assert short == "2 objects were below Smallest object"
    panel.update_add_ui_state(candidate_ids=[1], candidate_sizes={1: 5},
                              notes=[("counts", short, False)])
    qapp.processEvents()
    assert "…" not in panel.add_notes_label.text()


# --------------------------------------------------------------------------- #
# U-M3 -- REVIEW's S1 follows the selection
# --------------------------------------------------------------------------- #
def test_review_s1_for_none_one_and_two_selected():
    assert AP.added_line(3, 90, n_selected=0).endswith(AP.ADD_S1_SELECT_ONE)
    assert AP.added_line(3, 90, n_selected=1) == AP.added_line(
        3, 90, has_selection=True)
    assert AP.added_line(3, 90, n_selected=2).endswith(
        AP.ADD_S1_SELECT_JUST_ONE)
    s = {"has_patch": True, "review": (3, 90, 2)}
    assert AP.s1_sentence(s).endswith(AP.ADD_S1_SELECT_JUST_ONE)


def test_every_selection_change_refreshes_s1(h):
    selected = []
    h.adv.get_selected_add_pieces = lambda: list(selected)
    h._add_review = True
    h._add_piece_ids = [5, 6]
    h._add_session = [{"ids": [5, 6], "sizes": {5: 10, 6: 20}}]
    h._set_selected_label_all = lambda pid: None
    selected[:] = [5]
    h._on_add_piece_selected(5)
    assert h.adv.last("status_text") == AP.added_line(2, 30, n_selected=1)
    selected[:] = [5, 6]
    h._on_add_piece_selected(None)
    assert h.adv.last("status_text").endswith(AP.ADD_S1_SELECT_JUST_ONE)
    selected[:] = []
    h._on_add_piece_selected(None)
    assert h.adv.last("status_text").endswith(AP.ADD_S1_SELECT_ONE)


def test_the_panel_reports_every_selection_change(panel, qapp):
    got = []
    panel.set_add_callbacks(on_select_piece=got.append)
    panel.update_add_ui_state(review_active=True, piece_ids=[5, 6],
                              piece_sizes={5: 10, 6: 20})
    lst = panel.add_pieces_list
    lst.item(0).setSelected(True)
    lst.item(1).setSelected(True)
    lst.clearSelection()
    assert got[0] == 5
    assert None in got[1:]
    assert not panel.add_refine_piece_btn.isEnabled()

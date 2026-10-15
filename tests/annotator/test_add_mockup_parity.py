"""The Add page against the mockup (B2 fix pass U-M5, U-M6 and the LOWs).

The design mockup beats the written spec for what is on screen: the amber membrane-only line
under the Markers summary, ADD_HELP_RUN before the run count, the chevron
summaries, the row labels ("Adding", "From", "Nuclei channel", "Channel",
"Wall type", "Size", "7 placed"), and an idle Cells page that shows only
the preset until a box exists.  Plus ADD_HELP_DARK on screen inside
Advanced (mockup 8) and the audit's LOWs.
"""

from __future__ import annotations

import inspect
import os

import pytest

pytest.importorskip("qtpy")

from packages.epicycle_annotator import add_preview as AP  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _items(run_ok=True, nuclei_ok=True):
    return [
        (AP.ADD_ITEM_RUN.format(name="r1") if run_ok else AP.ADD_ITEM_NO_RUN,
         run_ok, ("labels", "r1" if run_ok else None)),
        (AP.ADD_ITEM_NUCLEI if nuclei_ok else AP.ADD_ITEM_NUCLEI_GREY,
         nuclei_ok, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None)),
    ]


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
    w.refresh_add_image_combo(["DAPI", "membrane"])
    w.set_add_preset("cells")
    w.update_add_ui_state(marker_sources=_items(), voxel_size_known=True)
    yield w
    w.hide()
    w.setParent(None)


def _labels_in(widget):
    from qtpy.QtWidgets import QLabel

    return [lb.text() for lb in widget.findChildren(QLabel)]


# --------------------------------------------------------------------------- #
# (e) idle Cells: only the preset and its help
# --------------------------------------------------------------------------- #
def test_idle_cells_shows_only_the_preset(panel):
    assert panel._add_cells_container.isHidden()
    panel.update_add_ui_state(box_active=True)
    assert not panel._add_cells_container.isHidden()


# --------------------------------------------------------------------------- #
# (d) row labels
# --------------------------------------------------------------------------- #
def test_the_row_labels_are_the_mockups(panel):
    texts = _labels_in(panel._add_s3_container)
    for want in ("Adding", "From", "Nuclei channel", "Channel", "Wall type"):
        assert want in texts, want
    for gone in ("Preset", "Membrane", "Walls"):
        assert gone not in texts, gone
    assert "Size" in _labels_in(panel._add_box_container)
    panel.update_add_ui_state(click_count=7)
    assert panel.add_click_count_label.text() == "7 placed"


def test_the_box_row_fits_the_card(panel, qapp):
    panel.update_add_ui_state(box_active=True)
    qapp.processEvents()
    row = panel._add_box_container.minimumSizeHint().width()
    assert row <= 290, row


# --------------------------------------------------------------------------- #
# (a) (b) order inside Markers
# --------------------------------------------------------------------------- #
def _index_in_markers(panel, widget):
    lay = panel._add_markers_container.layout()
    for i in range(lay.count()):
        it = lay.itemAt(i)
        if it.widget() is widget:
            return i
        sub = it.layout()
        if sub is not None:
            for j in range(sub.count()):
                if sub.itemAt(j).widget() is widget:
                    return i
    raise AssertionError("not in the Markers container")


def test_amber_above_the_source_combo_and_help_before_the_count(panel):
    amber = _index_in_markers(panel, panel._add_membrane_only_label)
    combo = _index_in_markers(panel, panel.add_marker_source_combo)
    helpl = _index_in_markers(panel, panel._add_source_help)
    count = _index_in_markers(panel, panel.add_run_count_label)
    assert amber < combo < helpl < count


# --------------------------------------------------------------------------- #
# (c) the chevron summaries
# --------------------------------------------------------------------------- #
def test_the_summaries_are_named_constants():
    assert AP.markers_summary("nuclei_masks", channel="DAPI") == (
        "Markers: nuclei masks · DAPI")
    assert AP.markers_summary("paint", n_clicks=7) == (
        "Markers: clicks · 7 placed")
    assert AP.markers_summary("labels", n_run_markers=14) == (
        "Markers: another run · 14 markers")
    assert AP.markers_summary("labels", n_run_markers=1) == (
        "Markers: another run · 1 marker")
    assert AP.markers_summary("labels") == "Markers: another run"
    assert AP.walls_summary(channel="membrane", wall_kind="Membrane sheets",
                            prep=["even out depth"]) == (
        "Walls: membrane · Membrane sheets · even out depth")


def test_the_chevrons_carry_the_summaries(panel):
    panel.update_add_ui_state(box_active=True)
    panel.add_nuclei_channel_combo.setCurrentIndex(0)
    panel.add_wall_channel_combo.setCurrentIndex(1)
    mk = panel.add_markers_btn.property("_epicycle_chevron_title")
    wl = panel.add_walls_btn.property("_epicycle_chevron_title")
    assert mk == "Markers: nuclei masks · DAPI"
    assert wl.startswith("Walls: membrane · ")
    assert wl.endswith(" · even out depth")


# --------------------------------------------------------------------------- #
# U-M6, and the LOWs
# --------------------------------------------------------------------------- #
def test_the_dark_help_is_on_screen_in_advanced(panel):
    assert panel._add_dark_help.text() == AP.ADD_HELP_DARK
    assert not panel._add_dark_help.isHidden()


def test_dark_below_keeps_its_last_sentence():
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    assert ADD_HELP_LINES["add_dark_factor_sb"] == AP.ADD_HELP_DARK_BELOW
    assert "Lower it if dim cells come out empty." in AP.ADD_HELP_DARK_BELOW


def test_millions_never_reads_as_the_limit_when_over_it():
    assert AP.millions(4_000_001) == "4.1"
    assert AP.millions(4_000_000) == "4.0"
    assert AP.millions(1_048_576) == "1.0"
    assert AP.millions(10) == "0.1"
    assert "4.0" not in AP.over_limit_line((1, 1, 4_000_001))


def test_the_inline_sentences_are_named_constants():
    from napari_epicycle.annotator import curation_3d_widget as CW

    src = inspect.getsource(CW)
    for old in ("Refining cell", "Nothing to add yet — Preview first.",
                "Move the crosshairs, then recenter.",
                "No added cells to refine yet.",
                "No added cells to smooth yet."):
        assert old not in src, old
    for const in ("ADD_REFINING_OBJECT", "ADD_S1_NOTHING_YET",
                  "ADD_RECENTER_NO_CROSSHAIR", "ADD_NO_ADDED_TO_REFINE",
                  "ADD_NO_ADDED_TO_SMOOTH"):
        assert "cell" not in getattr(AP, const).lower().replace(
            "cells", ""), const


def test_the_drag_help_is_hidden_while_busy(panel):
    panel.update_add_ui_state(box_active=True)
    assert not panel._add_box_help.isHidden()
    panel.update_add_ui_state(busy=True)
    assert panel._add_box_help.isHidden()
    panel.update_add_ui_state(busy=False)
    assert not panel._add_box_help.isHidden()


def test_the_amber_line_only_when_both_automatic_sources_are_grey(panel):
    panel.update_add_ui_state(box_active=True, single_channel=True,
                              marker_sources=_items(run_ok=True,
                                                    nuclei_ok=False))
    panel.add_marker_source_combo.setCurrentIndex(2)
    assert panel._add_membrane_only_label.isHidden(), (
        "a usable run is an automatic source")
    panel.update_add_ui_state(marker_sources=_items(run_ok=False,
                                                    nuclei_ok=False))
    panel.add_marker_source_combo.setCurrentIndex(2)
    assert not panel._add_membrane_only_label.isHidden()


def test_the_amber_line_is_off_over_candidates_and_review(panel):
    """The take-off that replaced step (4) -- the
    membrane-only amber is a pre-result instruction, so it is hidden while
    candidates or REVIEW are on screen and back on Discard."""
    panel.update_add_ui_state(box_active=True, single_channel=True,
                              marker_sources=_items(run_ok=False,
                                                    nuclei_ok=False))
    panel.add_marker_source_combo.setCurrentIndex(2)
    assert not panel._add_membrane_only_label.isHidden()
    panel.update_add_ui_state(candidate_ids=[5], candidate_sizes={5: 9},
                              candidate_edge_ids=set())
    assert panel._add_membrane_only_label.isHidden()
    panel.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                              candidate_edge_ids=set())
    assert not panel._add_membrane_only_label.isHidden()
    panel.update_add_ui_state(review_active=True)
    assert panel._add_membrane_only_label.isHidden()
    panel.update_add_ui_state(review_active=False)

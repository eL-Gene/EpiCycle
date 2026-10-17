"""Step 2 look, fix pass 4: short-mode room.  Below the tall-mode height the
Training column descriptions are header tooltips and the Slices chips are
icon + count (the word in the tooltip).  Qt-backed: run on its own with
``-o addopts=""``."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                                        # noqa: E402
from qtpy.QtWidgets import QApplication, QScrollArea                 # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_step2_curation import _real_widget                        # noqa: E402
from _step2_dock_helpers import _in_dock, _open                    # noqa: E402

from napari_epicycle.hitl_training_annotation.ui import _step2_lists as L  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _short_training(qapp):
    # the real dock: the shell's chrome leaves ~270 px for the section; the
    # widget's own chrome is smaller, so shorten the area to match
    w = _real_widget(qapp)
    sa = _in_dock(qapp, w, 576, 470)
    _open(qapp, w, 2)
    w._curation_short = None
    w._apply_curation_room()
    assert w._curation_short is True
    qapp.processEvents()
    qapp.processEvents()
    return w, sa


def test_training_descriptions_become_header_tooltips_in_short_mode(qapp):
    w, sa = _short_training(qapp)
    staged, setup = w.training_staged_column, w.training_setup_column
    for col in (staged, setup):
        assert col.desc_label.isHidden()
        assert col.desc_label.text() in col.title_label.toolTip()
    sa.close()


def test_staged_column_fits_without_scrolling_in_short_mode(qapp):
    """The viewport itself is the section room minus the foot (~137 px at the
    real 576 dock: the room is ~263 px, the foot ~108); what the descriptions
    freed is CONTENT: the staged list keeps >= 60 px (~80 live) and the column no longer
    scrolls."""
    w, sa = _short_training(qapp)
    w.training_panel.foot.set_stage_line(
        "Ready · Train writes Cell_lateral_round_03 from 14 lateral pairs "
        "(run Cell). Cancel works while a round trains.")
    qapp.processEvents()
    qapp.processEvents()
    left = w.training_panel.columns_splitter.widget(0)
    assert left.verticalScrollBar().maximum() == 0
    assert w.data_to_add_list.height() >= 60, w.data_to_add_list.height()
    sa.close()


def test_tall_mode_brings_the_training_descriptions_back(qapp):
    w, sa = _short_training(qapp)
    w.training_panel.set_short_mode(False)
    qapp.processEvents()
    for col in (w.training_staged_column, w.training_setup_column):
        assert not col.desc_label.isHidden()
        assert col.title_label.toolTip() == ""
    sa.close()


def test_staged_tooltip_follows_the_bucket(qapp):
    w, sa = _short_training(qapp)
    col = w.training_staged_column
    col.set_desc("axial")
    assert "axial" in col.title_label.toolTip()
    sa.close()


def test_slice_chips_are_icon_and_count_in_short_mode(qapp):
    w, sa = _short_training(qapp)
    recs = {f"P_XY_{i}": {"bucket": "lateral", "status": "predicted",
                          "patch_id": "P"} for i in range(7)}
    w._refresh_slice_chips(recs, {"bucket": "lateral", "patch": "P"})
    assert w.filter_predicted_btn.text() == "7"
    assert "Predicted" in w.filter_predicted_btn.toolTip()
    assert w.filter_all_btn.text() == "All 7"
    # tall mode: the words are back
    w._apply_chip_mode(False)
    assert w.filter_predicted_btn.text() == "Predicted 7"
    sa.close()


def test_the_five_chips_fit_one_row_in_short_mode(qapp):
    w = _real_widget(qapp)
    sa = _in_dock(qapp, w, 576, 470)
    _open(qapp, w, 1)
    w._curation_short = None
    w._apply_curation_room()
    recs = {f"P_XY_{i}": {"bucket": "lateral", "status": s, "patch_id": "P"}
            for i, s in enumerate(["predicted"] * 10 + ["curating", "staged"])}
    w._refresh_slice_chips(recs, {"bucket": "lateral", "patch": "P"})
    qapp.processEvents()
    qapp.processEvents()
    ys = {b.geometry().y() for b in (
        w.filter_all_btn, w.filter_predicted_btn, w.filter_curating_btn,
        w.filter_accepted_btn, w.filter_staged_btn)}
    assert len(ys) == 1, ys
    sa.close()

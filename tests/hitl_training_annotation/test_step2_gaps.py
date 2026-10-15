"""Step 2 Curation: the splitter gaps are Step 3's (6 | handle 6 | 8 across the
vertical splitter; the Slices header clears the horizontal handle).  Qt-backed:
run on its own with ``-o addopts=""``."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                                         # noqa: E402
from qtpy.QtWidgets import QApplication                               # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_step2_curation import _real_widget                          # noqa: E402
from test_step2_fix_pass3 import _in_dock, _open                      # noqa: E402

from napari_epicycle.fusion_reconstruction import ui as S3            # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _gaps(qapp, width, height):
    w = _real_widget(qapp)
    sa = _in_dock(qapp, w, width, height)
    _open(qapp, w, 1)
    w._curation_short = None
    w._apply_curation_room()
    for _ in range(4):
        qapp.processEvents()
    sp = w.curation_splitter
    h = sp.handle(1)
    hx0 = h.mapTo(w, h.rect().topLeft()).x()
    hx1 = hx0 + h.width()
    pl = w.train_patch_list
    left_edge = pl.mapTo(w, pl.rect().topRight()).x() + 1
    rc = w.curation_right_column
    vs = rc.view_switch
    right_edge = vs.mapTo(w, vs.rect().topLeft()).x()
    vh = w.curation_left_splitter.handle(1)
    vy1 = vh.mapTo(w, vh.rect().bottomLeft()).y() + 1
    vy0 = vh.mapTo(w, vh.rect().topLeft()).y()
    sl = w.segment_slice_data_list
    hdr = w.slices_header_patch
    hdr_top = hdr.parentWidget().mapTo(w, hdr.parentWidget().rect().topLeft()).y()
    # the header row's top = the slice pane's first layout item
    pane = w.curation_left_splitter.widget(1)
    row = pane.layout().itemAt(0).geometry()
    y_hdr = pane.mapTo(w, row.topLeft()).y()
    pat = w.curation_left_splitter.widget(0)
    pl_bottom = pl.mapTo(w, pl.rect().bottomLeft()).y() + 1
    gaps = dict(left=hx0 - left_edge, handle=h.width(), right=right_edge - hx1,
                below=y_hdr - vy1, above=vy0 - pl_bottom)
    return w, sa, gaps


def test_vertical_splitter_gaps_are_step3s_in_a_tall_dock(qapp):
    w, sa, g = _gaps(qapp, 1000, 900)
    assert g["handle"] == S3.PR_SPLIT_HANDLE == 6, g
    assert g["left"] == S3.PR_LEFT_INNER_MARGIN == 6, g
    assert g["right"] == S3.PR_RIGHT_INNER_MARGIN == 8, g
    sa.close()


def test_vertical_splitter_gaps_are_step3s_in_a_short_dock(qapp):
    w, sa, g = _gaps(qapp, 576, 487)
    assert (g["left"], g["handle"], g["right"]) == (6, 6, 8), g
    sa.close()


def test_the_slices_header_clears_the_horizontal_handle_in_a_tall_dock(qapp):
    w, sa, g = _gaps(qapp, 1000, 900)
    assert g["below"] == g["above"] == 4, g
    sa.close()


def test_a_short_dock_keeps_its_vertical_room(qapp):
    w, sa, g = _gaps(qapp, 576, 487)
    assert g["below"] == g["above"] == 0, g     # the budget is spent
    sa.close()


def test_the_dock_gets_no_horizontal_scroll_at_578(qapp):
    w, sa, g = _gaps(qapp, 578, 900)
    assert not sa.horizontalScrollBar().isVisible()
    sa.close()


def test_the_busy_bar_keeps_the_full_foot_width_while_busy(qapp):
    w, sa, g = _gaps(qapp, 1000, 900)
    f = w.curation_foot
    f.set_busy("[PATCH] Loading 'x' ...")
    for _ in range(3):
        qapp.processEvents()
    inner = f.width() - f.layout().contentsMargins().left()         - f.layout().contentsMargins().right()
    assert f.busy_bar.width() == inner, (f.busy_bar.width(), inner)
    assert f.busy_bar.toolTip() == "Loading 'x' ..."
    assert not f.busy_label.isVisible()
    row = f.busy_bar.parentWidget().layout()
    sibs = [w2 for w2 in f.findChildren(type(f.busy_label))
            if w2 is not f.busy_label and w2.isVisible()
            and abs(w2.mapTo(f, w2.rect().center()).y()
                    - f.busy_bar.mapTo(f, f.busy_bar.rect().center()).y()) < 4
            and w2 is not f.stage_label]
    assert sibs == [], sibs
    sa.close()

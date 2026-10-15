"""S1 4.3(a)(b): the " · measured at a × b × c µm" tail on Session and
Curation rows, and the Extract Features dialog's read-only voxel-size line."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _step3_pr_text as T  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_rows as R  # noqa: E402

CUR = {"zyx": [2.0, 0.5, 0.5], "source": "user"}
OLD = {"zyx": [1.0, 0.25, 0.25]}


def test_the_tail_text():
    assert T.measured_tail(OLD) == " · measured at 1.00 × 0.25 × 0.25 µm"
    assert T.measured_tail(None) == ""


def test_the_tail_compares_against_the_size_in_force():
    assert T.measured_differs(CUR, OLD)
    assert not T.measured_differs(CUR, {"zyx": [2.0, 0.5, 0.5]})
    assert not T.measured_differs(CUR, {"zyx": [2.0004, 0.5, 0.5]})
    assert not T.measured_differs(CUR, None)
    # audit H2: the size in force is the patch's own, else the fallback; a
    # patch measured at the fallback with no size of its own has no tail
    assert not T.measured_differs({"zyx": [1.0, 0.25, 0.25]}, OLD)
    # measured in voxels (None) says nothing; measured, but now in voxels
    assert T.measured_differs(None, OLD)


def test_session_row_gets_the_tail_in_warn():
    rec = R.build_session_row("a.tif", "reconstructed", "ZYX",
                              spacing=CUR, measured_spacing=OLD)
    assert rec["line3"] == "2.00 × 0.50 × 0.50 µm (set by you)"
    assert rec["line3_tail"] == T.measured_tail(OLD)
    assert rec["row_tip"].endswith("measured at 1.00 × 0.25 × 0.25 µm")
    same = R.build_session_row("a.tif", "reconstructed", "ZYX",
                               spacing=CUR)
    assert "line3_tail" not in same
    assert "line3_tail" not in R.build_session_row("a.tif", "reconstructed")


def test_curation_row_gets_the_tail_on_line_3():
    rec = R.build_curation_row("a.tif", "curating", qc_extracted=True,
                               spacing=CUR, measured_spacing=OLD)
    assert rec["line3_tail"] == T.measured_tail(OLD)
    assert rec["line3_tail"].startswith(T.S3_LINE_SEP)
    assert "line3_tail" not in R.build_curation_row(
        "a.tif", "curating", spacing=CUR)
    # no voxel line was asked for -> nothing to hang a tail on
    assert "line3_tail" not in R.build_curation_row(
        "a.tif", "curating", measured_spacing=OLD)
    # None = no voxel size: A3's amber "voxels" line, plus the tail
    none = R.build_curation_row("a.tif", "curating", spacing=None,
                                measured_spacing=OLD)
    assert none["line3_warn"] and none["line3_tail"]


def test_the_dialog_line_in_its_states():
    one = T.extract_spacing_lines({"a": CUR})
    assert one == [("This patch: 2.00 × 0.50 × 0.50 µm (set by you) — used "
                    "for the next extraction", False)]
    assert T.extract_spacing_lines({"a": None}) == [
        (T.S3_EXTRACT_SPACING_NONE, True)]
    many = T.extract_spacing_lines({"a": CUR, "b": None, "c": CUR})
    assert many == [("3 patches: 2 with their own voxel size, 1 use the "
                     "fallback below", True)]
    assert T.extract_spacing_lines({"a": CUR, "b": CUR})[0][1] is False
    assert T.extract_spacing_lines({}) == []


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_dialog_draws_the_line(app):
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )
    plain = QCExtractionDialog(None)
    assert plain.spacing_patch_label.isHidden()
    d = QCExtractionDialog(None, patch_spacing_lines=[("Hello µm", False)])
    assert not d.spacing_patch_label.isHidden()
    assert d.spacing_patch_label.text() == "Hello µm"
    w = QCExtractionDialog(None, patch_spacing_lines=[("Warn", True)])
    assert R.WARN_COLOUR in w.spacing_patch_label.styleSheet()
    d.set_patch_spacing_lines([])
    assert d.spacing_patch_label.isHidden()


def test_the_longest_dialog_line_fits_in_three_lines_at_480(app):
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )
    d = QCExtractionDialog(None, patch_spacing_lines=[
        (T.S3_EXTRACT_SPACING_NONE, True)])
    d.resize(480, 700)
    d.show()
    lab = d.spacing_patch_label
    for _ in range(4):
        app.processEvents()
    assert lab.width() > 150
    # the label's real height holds every wrapped line (no clipping) ...
    assert lab.height() >= lab.heightForWidth(lab.width()) - 2
    # ... and the sentence takes at most five offscreen lines (three with real fonts)
    lines = lab.heightForWidth(lab.width()) / lab.fontMetrics().lineSpacing()
    assert lines <= 5.01, lines      # offscreen glyphs run wide
    d.close()


def test_m10_a_tail_that_would_be_cut_gets_its_own_line(app):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    rec = R.build_session_row("p.tif", "reconstructed", "ZYX",
                              spacing=CUR, measured_spacing=OLD)
    delegate = D.StackRowDelegate(D.MODE_WIDE)
    wide = delegate.layout(QRect(0, 0, 3000, 0), rec, QFont())
    narrow = delegate.layout(QRect(0, 0, 300, 0), rec, QFont())
    assert "l3tail" not in wide["lines"]
    assert "l3tail" in narrow["lines"]
    # and the full sentence is on the row's tooltip either way
    assert "measured at 1.00 × 0.25 × 0.25 µm" in rec["row_tip"]

"""The QC Settings dialog's Extraction parameters each carry one dim line
under them saying what they do, in the field column, with the sentence
from the text module (the Features guide quotes it)."""
from __future__ import annotations

import pytest

from napari_epicycle.fusion_reconstruction import _step3_pr_text as T

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _dialog(app, runs=()):
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )

    d = QCExtractionDialog(None, match_runs=list(runs), match_patch="p")
    d.show()
    for _ in range(3):
        app.processEvents()
    return d


def _x(dlg, w) -> int:
    return w.mapTo(dlg, w.rect().topLeft()).x()


def _y(dlg, w) -> int:
    return w.mapTo(dlg, w.rect().topLeft()).y()


HINTS = (
    ("erosion_hint", "QC_EROSION_HINT", "erosion_iters_sb"),
    ("radius_hint", "QC_RADIUS_HINT", "max_radius_sb"),
    ("exclude_border_hint", "QC_EXCLUDE_BORDER_HINT", "exclude_border_cb"),
    ("invert_z_hint", "QC_INVERT_Z_HINT", "invert_z_cb"),
)


@pytest.mark.parametrize("attr,const,control", HINTS)
def test_each_parameter_has_its_line_under_it_in_the_field_column(
        app, attr, const, control):
    dlg = _dialog(app)
    hint = getattr(dlg, attr)
    ctl = getattr(dlg, control)
    assert hint.text() == getattr(T, const)
    assert hint.wordWrap() and not hint.isHidden()
    # indented to the field column: same left edge as its control, under it
    assert _x(dlg, hint) == _x(dlg, ctl)
    assert _y(dlg, hint) > _y(dlg, ctl)
    dlg.close()


def test_the_compare_with_line_is_indented_like_the_others(app):
    dlg = _dialog(app, [("nuclei", "Nuclei")])
    assert _x(dlg, dlg.match_hint) == _x(dlg, dlg.match_combo)
    assert _x(dlg, dlg.match_hint) == _x(dlg, dlg.erosion_hint)
    dlg.close()


def test_the_sentences_are_plain_and_short():
    for name in ("QC_EROSION_HINT", "QC_RADIUS_HINT",
                 "QC_EXCLUDE_BORDER_HINT", "QC_INVERT_Z_HINT"):
        s = getattr(T, name)
        assert s.endswith(".") and "{" not in s and len(s) < 160

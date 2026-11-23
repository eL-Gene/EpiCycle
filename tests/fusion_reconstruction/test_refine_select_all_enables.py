"""After All / None (the buttons and Ctrl+A) in Model Refinement the gates
follow the ticks: Extract is live with a tick and grey without."""
from __future__ import annotations

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as Ctl,
)


@pytest.fixture(scope="module")
def ui():
    app = QApplication.instance() or QApplication([])
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    yield w
    w.deleteLater()
    app.processEvents()


def _ctl(ui):
    c = Ctl.__new__(Ctl)
    c.ui = ui
    for flag in ("_step3_locked", "_refinement_running", "_extraction_running"):
        setattr(c, flag, False)
    c._count_pairs_on_disk = lambda: (4, 0, 1)
    c._load_extraction_manifest = lambda: {}
    c._set_refinement_status = lambda *a, **k: None
    c.working_dir = ""
    return c


def _rows(ui):
    ui.refresh_refinement_patch_list(
        {"A.tif": "finalized", "B.tif": "finalized"},
        {"A": {"spatial": 2, "axial": 2}, "B": {"spatial": 2, "axial": 2}})
    ui.set_refine_ticks(False)


def test_all_and_none_buttons_move_the_extract_gate(ui):
    c = _ctl(ui)
    _rows(ui)
    c._on_refine_panel_inputs_changed()
    assert ui.extract_training_pairs_btn.isEnabled() is False
    c._on_refine_select_all()
    assert ui.extract_training_pairs_btn.isEnabled() is True
    c._on_refine_deselect_all()
    assert ui.extract_training_pairs_btn.isEnabled() is False


def test_ctrl_a_in_the_list_moves_the_extract_gate(ui):
    """The controller's own wiring (not a hand-made connection) follows
    the list's Select-all request."""
    c = _ctl(ui)
    c.viewer = None
    _rows(ui)
    try:
        c._connect_signals()
    except AttributeError:
        pass        # a bare controller lacks later wiring; the list's came first
    c._on_refine_panel_inputs_changed()
    assert ui.extract_training_pairs_btn.isEnabled() is False
    ui.refine_patch_list.select_all_requested.emit()
    assert ui.extract_training_pairs_btn.isEnabled() is True

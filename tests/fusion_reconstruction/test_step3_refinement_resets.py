"""Model Refinement Brick A, I10 / L9 / L10: resets on a run switch and a
workspace browse, and the re-routed pack status writes."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from napari_epicycle.fusion_reconstruction import controller as cmod
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


def _bare():
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.ui = MagicMock()
    ctrl._last_refined_model_dir = "C:/old/run/models/refined_01"
    return ctrl


def test_reset_clears_the_loss_plot_and_the_last_model():
    ctrl = _bare()
    ctrl._reset_workspace_scoped_caches()
    ctrl.ui.refinement_loss_plot.clear_all.assert_called_once_with()
    assert ctrl._last_refined_model_dir is None


def test_reset_survives_a_ui_without_the_plot():
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.ui = SimpleNamespace()
    ctrl._last_refined_model_dir = "x"
    ctrl._reset_workspace_scoped_caches()
    assert ctrl._last_refined_model_dir is None


def test_adopt_workspace_runs_the_reset():
    src = inspect.getsource(FusionReconstructionController._adopt_workspace_quiet)
    assert "_reset_workspace_scoped_caches()" in src


def test_run_switch_and_layout_convert_run_the_reset():
    src = inspect.getsource(cmod.FusionReconstructionController)
    # run-switch orchestrator, layout conversion and adopt / browse
    assert src.count("self._reset_workspace_scoped_caches()") >= 3


def test_use_for_reprediction_refusal_goes_to_the_status_label():
    ctrl = _bare()
    ctrl._last_refined_model_dir = None
    ctrl._append_log = lambda _m: None
    ctrl._use_for_re_prediction()
    ctrl.ui.show_refine_status.assert_called_with(
        "No refined model available \u2014 run fine-tuning first.", "warn"
    )
    ctrl.ui.refine_pack_status_label.setText.assert_not_called()


def test_register_model_pack_is_gone():
    assert not hasattr(FusionReconstructionController, "_register_model_pack")


def test_no_pack_label_writes_left_in_the_controller():
    assert "refine_pack_status_label" not in inspect.getsource(cmod)

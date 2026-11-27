"""Step 3 names its own run "Step 3 run" wherever it shows a run's name.

A new project's Step 3 run is called "Default", and Step 2 has runs of its
own; a bare "Run: Default" read as the Step 2 run.  The run row of Predict +
Reconstruct, Model Refinement's run line and the console line of a run with
no channels saved say "Step 3 run".  Pure constants: no Qt.
"""
from __future__ import annotations

from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
from napari_epicycle.fusion_reconstruction import _step3_pr_text as P


def test_the_run_row_label():
    assert P.S3_RUN_LABEL == "Step 3 run:"


def test_model_refinements_run_line():
    assert M.MR_RUN_HEAD == "Step 3 run:"
    # the whole line is the head, the name and the tail, as the widget joins
    assert M.MR_RUN_LINE == " ".join(
        (M.MR_RUN_HEAD, M.MR_RUN_NAME, M.MR_RUN_TAIL.strip()))
    assert M.MR_RUN_LINE.format(name="Default") == (
        "Step 3 run: ● Default · trained models are kept with this run")


def test_the_console_line_of_a_run_with_no_channels_saved():
    assert P.S3_LOG_SEED_HEAD.format(run="Default").startswith(
        "[SETTINGS] Step 3 run Default: ")

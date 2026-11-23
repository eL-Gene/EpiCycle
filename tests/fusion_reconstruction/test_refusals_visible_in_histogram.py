"""A refusal must be readable with the plot area in its default Histogram
view: it goes to the pinned status slot, not to the card (hidden there)."""
from __future__ import annotations

import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _qc_plot_text as T
from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)

from .test_qc_plot_area import _load, _pump, area  # noqa: F401


def _ctrl(area):
    c = C.__new__(C)
    c.ui = type("U", (), {})()
    c.ui.qc_plot_area = area
    c.ui.qc_plot_widget = area.qc_plot_widget
    c._append_log = lambda t: None
    c._pr_status = lambda *a, **k: None
    c._qc_features_df = pd.DataFrame({"label": [1]})
    return c


def _shown(area):
    _pump()
    assert area.current_view() == "histogram"
    return area.status_label.text()


def test_mark_refusal_is_in_the_status_slot(area):
    _load(area)
    c = _ctrl(area)
    c._commit_triage_in_flight = True
    assert c._on_annotator_mark_tier(1, "good") is False
    assert _shown(area) == T.CARD_WARN_MARK_WHILE_SAVING


def test_extract_refusal_is_in_the_status_slot(area):
    _load(area)
    c = _ctrl(area)
    c._curation_dirty = True
    c._extract_features_current()
    assert _shown(area) == T.CARD_WARN_EXTRACT_UNSAVED


def test_other_patch_refusal_is_in_the_status_slot(area):
    _load(area)
    c = _ctrl(area)
    c._qc_table_patch = "P1.tif"
    c._qc_view_patch_name = "P2.tif"
    c._save_tiers_csv()
    assert _shown(area) == P.S3_SAVE_REFUSED_OTHER_PATCH.format(
        patch="P1.tif")


def test_save_failed_is_in_the_status_slot(area):
    from napari_epicycle.fusion_reconstruction.controller import (
        _report_save_failed,
    )
    _load(area)
    _report_save_failed(_ctrl(area))
    assert _shown(area) == P.S3_SAVE_FAILED


def test_step_exit_staged_refusal_is_in_the_status_slot(area):
    _load(area)
    c = _ctrl(area)
    c._staged_annotator_work = lambda: True
    assert c.on_step_exit() is False
    assert _shown(area) == P.S3_STAGED_WORK_OPEN

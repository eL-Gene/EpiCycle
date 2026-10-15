"""10 (brick P2 audit LOW-17): the column
``ui.py`` builds around the QC plot area, idle with no patch."""

from __future__ import annotations

from tests.fusion_reconstruction import _qc_plot_snap as S


def test_the_idle_busy_bar_says_nothing_and_the_row_is_grey():
    S.app()
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )

    w = FusionReconstructionCuration(viewer=None)
    bar = w.curation_busy_bar
    assert bar.text() == ""                  # never "0%"
    assert w.qc_plot_area.see_label.text() == T.SEE_NO_PATCH
    assert not w.commit_triage_btn.isEnabled()
    assert not w.assign_tier_btn.isEnabled()
    assert not w.tier_assign_combo.isEnabled()
    assert w.qc_summary_label.text() == T.QC_SUMMARY_NONE
    assert w.curation_status_label.text() == T.CURATION_STATUS_NONE
    w.deleteLater()

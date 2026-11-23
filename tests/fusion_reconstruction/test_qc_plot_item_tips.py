"""Every item of the plot's X, Y and Colour by lists says what it is.

Step 3 builds the plot with its words plus a ``describe`` function: a
measured feature's item carries "{column}: {description}" (the sentence the
QC Extraction window shows), each categorical Colour by entry its own
sentence, and an item with no sentence carries no tooltip at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fusion_reconstruction import _qc_plot_snap as S

CATEGORICAL = ("QC status", "QC tier", "Cluster", "Patch", "Border cell",
               "Topology")


def _df() -> pd.DataFrame:
    n = 8
    rng = np.random.RandomState(5)
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "area": rng.rand(n) * 1000 + 100,
        "solidity": rng.rand(n),
        "qc_score": rng.rand(n),
        "ch0_intensity_mean": rng.rand(n) * 50,
        "ch0_boundary_intensity_mean": rng.rand(n) * 50,
        "ch1_interior_intensity_mean": rng.rand(n) * 50,
        "mystery_metric": rng.rand(n),
        "qc_tier": ["good"] * n,
        "cluster_id": [0, 1] * (n // 2),
        "patch_name": ["p1"] * n,
        "is_border_cell": [0, 1] * (n // 2),
        "euler_number": np.ones(n, dtype=np.int64),
        "label_status": ["active"] * n,
        # what the old threshold filter wrote ("QC status" needs it)
        "qc_status": ["pass"] * n,
    })


def _plot(texts):
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME,
        QCPlotWidget,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    return QCPlotWidget(layout="workbench", texts=texts,
                        palette=STEP3_PLOT_PALETTE, tier_scheme=STEP3_SCHEME)


@pytest.fixture
def step3():
    from napari_epicycle.fusion_reconstruction._qc_plot_area import (
        _PLOT_TEXTS,
    )

    w = _plot(_PLOT_TEXTS)
    w.set_features(_df())
    yield w
    w.deleteLater()


def _tip(combo, text):
    from qtpy.QtCore import Qt

    i = combo.findText(text)
    assert i >= 0, text
    return combo.itemData(i, Qt.ItemDataRole.ToolTipRole)


def test_every_described_feature_carries_its_sentence(step3):
    from napari_epicycle.segmentation.analysis_widget import (
        _FEATURE_DESCRIPTIONS,
        _get_feature_description,
    )

    for combo in (step3._feature_x_combo, step3._feature_y_combo,
                  step3._color_combo):
        for col in ("area", "solidity", "qc_score"):
            assert _tip(combo, col) == f"{col}: {_FEATURE_DESCRIPTIONS[col]}"
        # a per-channel column reads its statistic's sentence
        desc = _get_feature_description("ch0_intensity_mean")
        assert desc
        assert _tip(combo, "ch0_intensity_mean") == (
            f"ch0_intensity_mean: {desc}")


def test_a_longer_channel_statistic_reads_its_own_sentence(step3):
    from napari_epicycle.segmentation.analysis_widget import (
        _CHANNEL_STAT_DESCRIPTIONS,
        _get_feature_description,
    )

    # "_intensity_mean" is also the end of these two: the longer suffix
    # wins
    assert _get_feature_description("ch0_boundary_intensity_mean") == (
        _CHANNEL_STAT_DESCRIPTIONS["_boundary_intensity_mean"])
    assert _get_feature_description("ch1_interior_intensity_mean") == (
        _CHANNEL_STAT_DESCRIPTIONS["_interior_intensity_mean"])
    assert _get_feature_description("ch0_intensity_mean") == (
        _CHANNEL_STAT_DESCRIPTIONS["_intensity_mean"])
    assert _tip(step3._feature_x_combo, "ch0_boundary_intensity_mean") == (
        "ch0_boundary_intensity_mean: "
        + _CHANNEL_STAT_DESCRIPTIONS["_boundary_intensity_mean"])


def test_qc_status_is_offered_only_with_its_column(step3):
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T

    assert _tip(step3._color_combo, "QC status") == (
        T.COLOUR_BY_QC_STATUS_TIP)
    assert T.COLOUR_BY_QC_STATUS_TIP == (
        "Whether each label passed the old threshold filter: pass, "
        "flagged, or excluded as too small. That filter is no longer in "
        "the panel, so a new table shows every label grey; colour by QC "
        "tier instead.")
    step3.set_features(_df().drop(columns=["qc_status"]))
    assert step3._color_combo.findText("QC status") < 0
    assert step3._color_combo.itemText(0) == "QC tier"


def test_a_column_with_no_description_carries_no_tooltip(step3):
    for combo in (step3._feature_x_combo, step3._feature_y_combo,
                  step3._color_combo):
        assert _tip(combo, "mystery_metric") is None


def test_the_six_group_entries_carry_their_sentences(step3):
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T

    assert tuple(T.COLOUR_BY_TIPS) == CATEGORICAL
    for entry in CATEGORICAL:
        assert _tip(step3._color_combo, entry) == T.COLOUR_BY_TIPS[entry]
    assert T.COLOUR_BY_TIPS["QC tier"] == T.COLOUR_BY_QC_TIER_TIP
    assert T.COLOUR_BY_TIPS["Topology"] == T.COLOUR_BY_TOPOLOGY_TIP


def test_the_tips_follow_a_new_table(step3):
    df = _df().drop(columns=["area"]).assign(sphericity=0.5)
    step3.set_features(df)
    assert step3._feature_x_combo.findText("area") < 0
    assert _tip(step3._feature_x_combo, "sphericity").startswith(
        "sphericity: ")
    assert _tip(step3._color_combo, "Cluster")


def test_a_host_without_the_words_sets_no_tooltip():
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )

    texts = {k: v for k, v in STEP3_PLOT_TEXTS.items()
             if k != "COLOUR_BY_TIPS"}
    w = _plot(texts)
    try:
        w.set_features(_df())
        for combo in (w._feature_x_combo, w._color_combo):
            for i in range(combo.count()):
                assert _tip(combo, combo.itemText(i)) is None
    finally:
        w.deleteLater()


def test_a_failing_describer_never_breaks_the_load():
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )

    def boom(_name):
        raise RuntimeError("no")

    w = _plot({**STEP3_PLOT_TEXTS, "describe": boom})
    try:
        w.set_features(_df())
        assert _tip(w._feature_x_combo, "area") is None
        assert _tip(w._color_combo, "Patch") == (
            STEP3_PLOT_TEXTS["COLOUR_BY_TIPS"]["Patch"])
    finally:
        w.deleteLater()


def test_a_bad_tooltip_template_never_breaks_the_load():
    from napari_epicycle.fusion_reconstruction._qc_plot_area import (
        _PLOT_TEXTS,
    )

    w = _plot({**_PLOT_TEXTS, "FEATURE_TOOLTIP": "{column}: {missing}"})
    try:
        w.set_features(_df())
        assert w._feature_x_combo.findText("area") >= 0
        assert _tip(w._feature_x_combo, "area") is None
        assert _tip(w._color_combo, "Patch") == (
            _PLOT_TEXTS["COLOUR_BY_TIPS"]["Patch"])
    finally:
        w.deleteLater()

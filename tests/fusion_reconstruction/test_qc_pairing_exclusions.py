"""Where the match columns are kept out.

The five match columns compare the active run with another run of the
patch; they are not measurements of a cell.  So the bounded two are never
log-transformed, ``match_id`` (a label number of the other run) is never
an axis or a workbench feature, none of them joins the Standard PCA set,
and the "Overlap with other run" category stays out of the feature
manifest unless it is asked for -- Step 4's Measure dialog and Step 3's
checklist are unchanged.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.qc_pairing import COLUMNS

MATCH = list(COLUMNS)


def test_the_bounded_match_columns():
    from napari_epicycle.fusion_reconstruction.qc_features import (
        BOUNDED_FEATURES,
    )
    assert "match_overlap" in BOUNDED_FEATURES
    assert "match_minor_ratio" in BOUNDED_FEATURES
    for c in ("match_count", "match_id", "match_shared"):
        assert c not in BOUNDED_FEATURES


def test_the_category_descriptions_and_order():
    from napari_epicycle.segmentation.analysis_widget import (
        _CATEGORY_ORDER, _FEATURE_CATEGORIES, _FEATURE_DESCRIPTIONS,
        _MATCH_CATEGORY,
    )
    from napari_epicycle.fusion_reconstruction.qc_pairing_group import (
        LINK_FLOOR,
    )
    assert _MATCH_CATEGORY == "Overlap with other run"
    for c in MATCH:
        assert _FEATURE_CATEGORIES[c] == "Overlap with other run", c
        text = _FEATURE_DESCRIPTIONS[c]
        # one sentence each ("e.g." is not a full stop)
        assert text.replace("e.g. ", "").count(". ") == 0, c
        assert text.endswith("."), c
    # the link floor in use, written as the help writes it
    assert f"at least {LINK_FLOOR:.2f} of the smaller object" in (
        _FEATURE_DESCRIPTIONS["match_count"])
    assert _FEATURE_DESCRIPTIONS["match_id"].endswith(
        "(0 = none, too small, or not measured).")
    for c in ("match_count", "match_id"):
        assert "the other run" in _FEATURE_DESCRIPTIONS[c], c
    i = _CATEGORY_ORDER.index("Topology")
    assert _CATEGORY_ORDER[i + 1] == "Overlap with other run"


def test_the_manifest_leaves_the_match_category_out_by_default():
    from napari_epicycle.segmentation.analysis_widget import (
        _CATEGORY_ORDER, _FEATURE_CATEGORIES,
    )
    from napari_epicycle.segmentation.feature_registry import (
        get_feature_manifest,
    )
    m = get_feature_manifest()
    assert "Overlap with other run" not in m
    assert not any(f.startswith("match_") for v in m.values() for f in v)
    # the same manifest as before the category existed
    expected = {}
    for cat in _CATEGORY_ORDER:
        if cat == "Overlap with other run":
            continue
        feats = [f for f, c in _FEATURE_CATEGORIES.items() if c == cat]
        if feats:
            expected[cat] = feats
    assert m == expected
    assert list(m) == list(expected)
    with_ch = get_feature_manifest(
        checked_channels=[{"index": 0, "prefix": "ch0_"}])
    assert "Overlap with other run" not in with_ch


def test_the_manifest_includes_the_match_category_on_request():
    from napari_epicycle.segmentation.feature_registry import (
        get_feature_manifest,
    )
    m = get_feature_manifest(include_match=True)
    assert m["Overlap with other run"] == MATCH
    keys = list(m)
    assert keys.index("Overlap with other run") == keys.index("Topology") + 1


def test_the_dialogs_built_from_the_default_manifest_list_no_match_column(
        app):
    # both controllers build their window from get_feature_manifest() with
    # no include_match (Step 3 with checked_channels=None, Step 4 with the
    # channel boxes): neither window then offers a match column
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )
    from napari_epicycle.fusion_reconstruction.qc_features import (
        PCA_PRESETS,
    )
    from napari_epicycle.segmentation.extract_dialog import (
        ExtractFeaturesDialog,
    )
    from napari_epicycle.segmentation.feature_registry import (
        get_feature_manifest,
    )

    s3 = QCExtractionDialog(
        None, feature_manifest=get_feature_manifest(checked_channels=None))
    s4 = ExtractFeaturesDialog(
        None, n_labels=10, pca_presets=PCA_PRESETS,
        feature_manifest=get_feature_manifest(
            checked_channels=[{"index": 0, "prefix": "ch0_"}]))
    try:
        for dlg in (s3, s4):
            offered = {f for gw in dlg._group_widgets.values()
                       for f in getattr(gw, "_checkboxes", {})}
            assert offered, type(dlg).__name__
            assert not offered & set(MATCH), type(dlg).__name__
    finally:
        s3.deleteLater()
        s4.deleteLater()


def test_the_standard_pca_set_has_no_match_column():
    from napari_epicycle.segmentation.extract_dialog import standard_pca_set
    feats = ({"area", "solidity", "centroid-0", "major_axis_z",
              "match_status"} | set(MATCH))
    assert standard_pca_set(feats) == {"area", "solidity"}
    # exactly the check's columns: another feature named "match_..." stays
    assert standard_pca_set({"area", "match_score_lab"}) == {
        "area", "match_score_lab"}


def test_the_standard_preset_in_the_dialog_uses_the_shared_set():
    from napari_epicycle.segmentation import extract_dialog as ED
    src = inspect.getsource(ED.ExtractFeaturesDialog._apply_preset_by_index)
    assert "standard_pca_set(self._all_features)" in src
    src = inspect.getsource(
        ED.ExtractFeaturesDialog._is_selection_matching_preset)
    assert "standard_pca_set(self._all_features)" in src


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("qtpy")
    from tests.fusion_reconstruction import _qc_plot_snap as S
    return S.app()


def _table(n=8):
    """A table with the five match columns and a feature whose name also
    starts "match_" but is not the check's."""
    rng = np.random.RandomState(4)
    df = pd.DataFrame({
        "label": np.arange(1, n + 1),
        "area": rng.rand(n) * 500 + 100,
        "solidity": rng.rand(n),
        "qc_score": rng.rand(n),
        "match_score_lab": rng.rand(n),
        "label_status": ["active"] * n,
    })
    df["match_count"] = [0, 1, 2, 1, 1, 0, 1, 1][:n]
    df["match_id"] = [0, 7, 8, 9, 9, 0, 3, 4][:n]
    df["match_overlap"] = rng.rand(n)
    df["match_shared"] = [0, 1, 1, 2, 2, 0, 1, 1][:n]
    df["match_minor_ratio"] = rng.rand(n)
    return df


def test_step3_select_all_skips_exactly_the_match_columns(app):
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )

    manifest = {"Shape": ["area", "solidity", "match_score_lab"],
                "Overlap with other run": list(MATCH) + ["match_status"]}
    dlg = QCExtractionDialog(None, feature_manifest=manifest)
    try:
        dlg._select_all_features()
        checked = {f for gw in dlg._group_widgets.values()
                   for f in gw.checked_features()}
        assert checked == {"area", "solidity", "match_score_lab"}
    finally:
        dlg.deleteLater()


def test_match_id_is_never_a_plot_axis(app):
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    w = QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS)
    try:
        w.set_features(_table())
        for combo in (w._feature_x_combo, w._feature_y_combo):
            items = {combo.itemText(i) for i in range(combo.count())}
            assert "match_id" not in items
            assert {"match_count", "match_overlap", "match_shared",
                    "match_minor_ratio", "match_score_lab"} <= items
    finally:
        w.deleteLater()


def test_match_id_is_never_a_histogram_axis(app):
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )

    p = TriageHistogramPanel()
    try:
        p.set_data(_table())
        combo = p._feature_combo
        items = {combo.itemText(i) for i in range(combo.count())}
        assert "match_id" not in items
        assert {"match_count", "match_overlap", "match_score_lab"} <= items
    finally:
        p.deleteLater()


def test_match_id_is_never_a_workbench_feature(app):
    from napari_epicycle.segmentation.analysis_widget import (
        AnalysisControlsWidget,
    )

    w = AnalysisControlsWidget()
    try:
        w.set_feature_data(_table(), exclude_border=False)
        assert "match_id" not in w._feature_names
        assert {"match_count", "match_overlap", "match_shared",
                "match_minor_ratio", "area"} <= set(w._feature_names)
    finally:
        w.deleteLater()

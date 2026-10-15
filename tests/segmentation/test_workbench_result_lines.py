"""18: every Workbench result line
is a named sentence, on its own label.

* An AST walk of the CONTROLLER refuses any string literal or f-string
  that reaches a Workbench status label (``AnalysisControlsWidget.
  STATUS_LABELS``) or one of the Workbench's text setters -- the
  widget's own module is held by ``test_workbench_sentences.py``.
* PCA_DONE / PCA_EXCLUDED on the PCA card, MAP_DONE on the Map tab's
  line, CLUSTER_DONE / CLUSTER_DONE_HDB / CLUSTER_NEEDS_UMAP on the
  Groups tab's line, MAP_AXES from ``set_map_input``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from napari_epicycle.segmentation import _workbench_text as T

from ._workbench_ctrl import pump, table, wired

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "napari_epicycle" / "segmentation" / "controller.py"

#: Every call that puts text on the Workbench.
TEXT_SETTERS = ("show_card_message", "show_map_result", "show_map_error",
                "show_groups_result", "set_busy_strip", "_card_message",
                "_wb_map_error", "_wb_groups_result", "set_workbench_lock",
                "set_map_running", "flash_busy_strip")


def _is_literal(node) -> bool:
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value != ""
    if isinstance(node, ast.BinOp):
        return _is_literal(node.left) or _is_literal(node.right)
    if isinstance(node, ast.IfExp):
        return _is_literal(node.body) or _is_literal(node.orelse)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"):
        # "...".format(...) is a literal; T.X.format(...) is not (E1-6)
        return _is_literal(node.func.value)
    return False


def _status_labels():
    from napari_epicycle.segmentation.analysis_widget import (
        AnalysisControlsWidget,
    )
    return set(AnalysisControlsWidget.STATUS_LABELS)


def _violations(source: str) -> list:
    tree = ast.parse(source)
    labels = _status_labels()
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)):
            continue
        attr = node.func.attr
        if attr == "setText" and node.args:
            owner = node.func.value
            if (isinstance(owner, ast.Attribute) and owner.attr in labels
                    and _is_literal(node.args[0])):
                bad.append(f"line {node.lineno}: {owner.attr}.setText")
        elif attr in TEXT_SETTERS and node.args:
            if _is_literal(node.args[0]):
                bad.append(f"line {node.lineno}: {attr}")
        elif attr == "show_selection":
            for kw in node.keywords:
                if kw.arg == "count_text" and _is_literal(kw.value):
                    bad.append(f"line {node.lineno}: show_selection")
    return bad


def test_no_literal_reaches_the_workbench_from_the_controller():
    bad = _violations(CTRL.read_text(encoding="utf-8"))
    assert not bad, bad


@pytest.mark.parametrize("src", [
    "aw.status_label.setText('Computing')",
    "aw.groups_result_label.setText(f'{k} groups')",
    "self._card_message('Marked {m}'.format(m=1))",
    "aw.show_selection(count_text='3 cells', mix=[], chips_html='')",
    "aw.set_busy_strip('busy' + x)",
])
def test_the_literal_test_has_teeth(src):
    """E1-6: each of these must be caught."""
    assert _violations(src)


@pytest.mark.parametrize("src", [
    "aw.status_label.setText(T_WB().MAP_DONE.format(map=w, input=i))",
    "self._card_message('')",
    "aw.set_busy_strip(None)",
])
def test_the_literal_test_lets_constants_through(src):
    assert not _violations(src)


@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(40))
    yield c, wb
    wb.deleteLater()
    pump()


def _land(c, outcome):
    c._eda_launch = {"kind": outcome["kind"], "token": 1,
                     "params": outcome.pop("params", {}),
                     "stack": c._current_stack_name,
                     "n": len(c._qc_features_df),
                     # launched from the plot on screen (X1-2)
                     "plot": c._pw.plot_type_combo.currentText()}
    c._on_eda_finished(dict(outcome, token=1))


def test_umap_line_and_axes_name_the_input(cw):
    c, wb = cw
    _land(c, {"kind": "umap", "projected": np.zeros((40, 2)),
              "input": T.INPUT_PCA.format(n=10), "log": ""})
    assert wb.status_label.text() == T.MAP_DONE.format(
        map="UMAP", input="PCA (10 components)")
    pw = wb.qc_plot_widget
    assert pw.plot_type_combo.currentText() == "UMAP"
    assert pw._wb_axes_note.text() == T.MAP_AXES.format(
        map="UMAP", input="PCA (10 components)")


def test_tsne_line(cw):
    c, wb = cw
    _land(c, {"kind": "tsne", "projected": np.zeros((40, 2)),
              "input": T.INPUT_FEATURES, "log": ""})
    assert wb.status_label.text() == T.MAP_DONE.format(
        map="t-SNE", input="the rescaled features")


def test_hdbscan_line_counts_noise(cw):
    c, wb = cw
    labels = np.array([0] * 20 + [1] * 15 + [-1] * 5)
    _land(c, {"kind": "cluster", "labels": labels, "algorithm": "hdbscan",
              "n_groups": 2, "noise": 5, "seed": 42, "input": "x",
              "log": ""})
    assert wb.groups_result_label.text() == T.CLUSTER_DONE_HDB.format(
        k=2, noise="5")
    assert "noise" in wb._groups_legend_label.text()


def test_cluster_from_a_missing_umap_says_so(cw):
    c, wb = cw
    c._on_compute_clusters({"input_space": "UMAP", "algorithm": "leiden"})
    assert wb.groups_result_label.text() == T.CLUSTER_NEEDS_UMAP
    assert not c._eda_running


def test_a_group_swatch_selects_that_group(cw):
    c, wb = cw
    labels = np.array([0] * 25 + [1] * 15)
    _land(c, {"kind": "cluster", "labels": labels, "algorithm": "kmeans",
              "n_groups": 2, "noise": 0, "seed": 1, "input": "x",
              "log": ""})
    wb._on_group_link("group://1")
    pump()
    assert sorted(c._selected_label_ids) == list(range(26, 41))
    assert wb.sel_msg_label.text() == T.CAT_PICKED.format(cat="G2", m="15")


def test_a_pca_line_is_pca_done(cw):
    c, wb = cw
    feats = [col for col in c._qc_features_df.columns
             if col not in ("label", "qc_tier", "is_border_cell")]
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_pca_full,
    )
    res = compute_pca_full(c._qc_features_df, n_components=3,
                           features=feats, exclude_border=True, mode="eda")
    _land(c, {"kind": "pca", "result": res, "log": "",
              "params": {"features": feats, "n_components": 3}})
    v = float(np.asarray(res["explained_variance_ratio"])[:2].sum()) * 100
    assert wb._pca_done_label.text() == T.PCA_DONE.format(
        f=len(feats), v=f"{v:.0f}")
    assert wb.status_label.text() == ""        # the PCA line is its own


def test_a_pca_made_from_other_ticks_lands_stale(cw):
    """The token names the ticked features; a PCA whose features
    were ticked away while it ran lands with PCA_STALE on (recorded)."""
    c, wb = cw
    feats = [col for col in c._qc_features_df.columns
             if col not in ("label", "qc_tier", "is_border_cell")]
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_pca_full,
    )
    res = compute_pca_full(c._qc_features_df, n_components=3,
                           features=feats, exclude_border=True, mode="eda")
    wb.restore_selected_features(feats[:3])          # ticked away since
    _land(c, {"kind": "pca", "result": res, "log": "",
              "params": {"features": feats, "n_components": 3}})
    assert not wb._pca_stale_label.isHidden()
    assert c._last_pca_projected is res["projected"]

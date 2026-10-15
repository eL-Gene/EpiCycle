"""The Workbench on a load, a Reset, a Remove
and a Measure.

The selection clears (SEL_NONE), the mark history clears, the map
results go (loadings table, result lines, Groups legend), a running map
is stopped with LOG_MAP_ABANDONED -- and the seed box keeps its value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.segmentation import _workbench_text as T

from ._workbench_ctrl import pump, select, table, wired


@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(30, tiers=[""] * 30))
    yield c, wb
    wb.deleteLater()
    pump()


def _dirty_workbench(c, wb):
    """A selection, a mark, a PCA card, a UMAP line, groups, a seed."""
    select(c, [1, 2])
    c._mark_bad()
    feats = [col for col in c._qc_features_df.columns
             if col not in ("label", "qc_tier", "is_border_cell")]
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_pca_full,
    )
    res = compute_pca_full(c._qc_features_df, n_components=3,
                           features=feats, exclude_border=True, mode="eda")
    wb.show_pca_result(res)
    wb.show_map_result(T.MAP_DONE.format(map="UMAP", input="x"))
    wb.set_maps_old(True)
    c._qc_features_df["cluster_id"] = np.arange(30) % 2
    wb.qc_plot_widget._df = c._qc_features_df
    wb.show_groups_result(T.CLUSTER_DONE.format(k=2, method="K-means",
                                                seed=5))
    wb.refresh_groups_legend()
    wb._cluster_seed_spin.setValue(5)
    assert wb.loadings_table.rowCount() > 0
    assert c._mark_history


def _assert_clean(c, wb):
    assert c._mark_history == []
    assert c._selected_label_ids == []
    assert wb.qc_selection_label.text() == T.SEL_NONE
    assert wb.sel_msg_label.text() == ""
    assert wb.loadings_table.rowCount() == 0
    assert wb._pca_results_panel.isHidden()
    assert wb.status_label.text() == ""
    assert wb._maps_old_label.isHidden()
    assert wb.groups_legend.isHidden()
    assert wb.groups_result_label.isHidden()
    assert wb._cluster_seed_spin.value() == 5          # the seed stays
    assert not wb.undo_mark_btn.isEnabled()


def test_remove_bad_cells(cw, monkeypatch):
    import napari_epicycle.ui_shell.theme as theme

    c, wb = cw
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    vol = np.zeros((2, 4, 4), np.uint32)
    vol.flat[:30] = np.arange(1, 31)
    c._labels_volume = vol
    _dirty_workbench(c, wb)
    c._delete_bad_labels()
    _assert_clean(c, wb)


def test_reset_and_unload(cw):
    c, wb = cw
    _dirty_workbench(c, wb)
    c._clear_workbench_caches()          # what Reset and an unload call
    _assert_clean(c, wb)


def test_measure(cw):
    c, wb = cw
    _dirty_workbench(c, wb)
    c._measure_launch = {"token": 1, "stack": "s.tif", "spacing": None,
                         "z_orientation": None}
    c._on_extract_finished(pd.DataFrame({"label": [1, 2, 3]}), "s.tif")
    # the marks are carried over by label id (F18) ...
    assert list(c._qc_features_df["qc_tier"]) == ["bad", "bad", ""]
    # ... but the card, the history and the map results start again
    assert c._mark_history == []
    assert wb.qc_selection_label.text() == T.SEL_NONE
    assert wb.loadings_table.rowCount() == 0


def test_load(cw, monkeypatch, tmp_path):
    """B1b's load (a row click on a worker): the Workbench clears at
    REQUEST time, before the worker is started (integration of B1b+B3b)."""
    c, wb = cw
    _dirty_workbench(c, wb)
    labels = tmp_path / "other_labels.tif"
    labels.write_bytes(b"x")
    monkeypatch.setattr(c, "_settle_unsaved", lambda what: True)
    monkeypatch.setattr(c, "_refuse_while_locked", lambda what: False)
    monkeypatch.setattr(c, "_ensure_writable_state", lambda: None)
    monkeypatch.setattr(c.state_manager, "get_record",
                        lambda name: {"labels_path": str(labels),
                                      "source_stack_path": ""})
    monkeypatch.setattr(c, "_resolve_source_path", lambda *a: "")
    started = []

    def fake_start(worker, flag, **kw):
        # the Workbench is already clean when the worker would start
        started.append(list(c._mark_history))
        return None

    monkeypatch.setattr(c, "_start_step4_thread", fake_start)

    class Item:
        def data(self, role):
            return "other.tif"

    class QCList:
        def selectedItems(self):
            return [Item()]

    c.ui.qc_stacks_list = QCList()
    c._load_reconstruction()
    assert started == [[]]
    assert c._mark_history == []
    assert wb.qc_selection_label.text() == T.SEL_NONE
    assert wb.loadings_table.rowCount() == 0


def test_a_load_stops_a_running_map(cw, monkeypatch):
    """The abandon rule, without a thread: the flag is up, the load
    stops it and the answer, when it comes, is dropped."""
    c, wb = cw
    c._eda_running = True
    c._eda_launch = {"kind": "tsne", "token": 3, "params": {},
                     "stack": "s.tif", "n": 30}
    c._workbench_cells_changed("load")
    assert c._eda_abandoned
    assert any(T.LOG_MAP_ABANDONED.format(map="t-SNE") in line
               for line in c.ui.log_output.lines)
    assert wb.busy_label.text() == T.STOPPING.format(map="t-SNE")
    c._on_eda_finished({"kind": "tsne", "token": 3,
                        "projected": np.zeros((30, 2)), "input": "x"})
    assert wb.qc_plot_widget._tsne_cache is None
    c._eda_running = False

"""The B3b re-audit's fixes (MED-1..3, LOW-1..6)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.segmentation import _eda_worker as EW
from napari_epicycle.segmentation import _workbench_text as T

from ._workbench_ctrl import app, pump, select, table, wired

TIERS = ["good"] * 5 + ["bad"] * 3 + ["garbage"] + ["review"] + [""] * 10


def _wait(pred, timeout=60.0):
    a = app()
    t0 = time.monotonic()
    while not pred():
        a.processEvents()
        time.sleep(0.01)
        if time.monotonic() - t0 > timeout:
            raise AssertionError("timed out")
    for _ in range(5):
        a.processEvents()


def _fake_umap(params, inputs):
    time.sleep(0.05)
    return {"projected": np.full((len(inputs["df"]), 2), 7.0),
            "input": T.INPUT_FEATURES, "log": ""}


def _umap_params():
    return {"features": ["x"], "transforms": {}, "scaling": "eda",
            "winsorize": None, "n_neighbors": 5, "min_dist": 0.1,
            "input_space": "Features", "pca_dims": 3, "metric": "euclidean"}


@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(20, tiers=TIERS))
    pw = wb.qc_plot_widget
    pw._color_combo.setCurrentText("QC tier")
    pw._redraw()
    pump()
    yield c, wb
    th = c._eda_thread
    if th is not None:
        _wait(lambda: th.isFinished())
    wb.deleteLater()
    pump()


# ── MED-1 ──

def test_the_warm_up_run_is_not_the_rate(cw, monkeypatch):
    c, wb = cw
    monkeypatch.setitem(EW._RUNNERS, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert wb.busy_label.text() == T.BUSY_WARMUP.format(map="UMAP")
    _wait(lambda: not c._eda_running)
    assert "umap" not in c._eda_rates          # the compile is not a rate
    c._on_compute_umap(_umap_params())
    # warmed: the spec's figure for 20 cells, not the warm-up's
    assert wb.busy_label.text() == T.BUSY_FEW.format(map="UMAP")
    _wait(lambda: not c._eda_running)
    assert "umap" in c._eda_rates


# ── MED-2 ──

def test_removed_cells_leave_the_legend_and_the_picks(cw, monkeypatch):
    import napari_epicycle.ui_shell.theme as theme

    c, wb = cw
    pw = wb.qc_plot_widget
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    vol = np.zeros((2, 4, 4), np.uint32)
    vol.flat[:20] = np.arange(1, 21)
    c._labels_volume = vol
    c._delete_bad_labels()                 # bad + garbage removed
    pw._color_combo.setCurrentText("QC tier")
    pw._redraw()
    html = pw._legend_label.text()
    assert "bad</a>&nbsp;<span style=\"color:#888888;\">0</span>" in html
    pw._toggle_category_selection("qc_tier", "bad")
    pump()
    assert c._selected_label_ids == []
    c._on_category_selected("qc_tier", "bad")
    assert c._selected_label_ids == []


# ── MED-3 ──

def test_unmark_help_says_unmarked():
    assert "unmarked" in T.WORKBENCH_HELP_LINES["Unmark"]
    assert "unassigned" not in T.WORKBENCH_HELP_LINES["Unmark"]


# ── LOW-1 ──

def test_the_flash_timer_is_parented_and_stopped(cw):
    c, wb = cw
    wb.flash_busy_strip(T.STOPPED.format(map="UMAP"), ms=10000)
    assert wb._flash_timer.parent() is wb and wb._flash_timer.isActive()
    wb.set_busy_strip(T.BUSY.format(map="UMAP", s=5))
    assert not wb._flash_timer.isActive()
    wb.flash_busy_strip(T.STOPPED.format(map="UMAP"), ms=10000)
    c._workbench_cells_changed("load")
    assert wb.busy_strip.isHidden()
    assert not wb._flash_timer.isActive()


# ── LOW-3 ──

def test_the_plot_words_come_in_through_texts(cw):
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_PLOT_TEXTS,
    )
    c, wb = cw
    pw = wb.qc_plot_widget
    for key in ("UNMARKED_WORD", "GROUP_WORD", "NOISE_WORD"):
        assert key in WORKBENCH_PLOT_TEXTS
    pw._wb_texts = dict(pw._wb_texts, UNMARKED_WORD="ZZ",
                        GROUP_WORD="K{n}", NOISE_WORD="NN")
    pw._redraw()
    assert "ZZ" in pw._legend_label.text()
    c._qc_features_df["cluster_id"] = np.array([-1] + [0] * 19)
    pw._df = c._qc_features_df
    if pw._color_combo.findText("Cluster") < 0:
        pw._color_combo.addItem("Cluster")
    pw._color_combo.setCurrentText("Cluster")
    pw._redraw()
    html = pw._legend_label.text()
    assert "K1" in html and "NN" in html


# ── LOW-4 ──

def test_a_kept_pick_keeps_its_positions_on_any_index(tmp_path):
    df = table(20, tiers=TIERS)
    df.index = np.arange(100, 120)
    c, wb = wired(tmp_path, df)
    pw = wb.qc_plot_widget
    pw._toggle_category_selection("qc_tier", "good")
    pump()
    pos = list(pw._selected_category_df_indices)
    assert pos == [0, 1, 2, 3, 4]
    pw.plot_type_combo.setCurrentText("PCA")
    pw.plot_type_combo.setCurrentText("Scatter")
    pump()
    assert list(pw._selected_category_df_indices) == pos
    assert set(np.where(pw._get_selected_category_mask())[0]) == set(pos)
    wb.deleteLater()
    pump()


# ── LOW-5 ──

def test_a_map_that_did_not_take_the_plot_says_where_it_is(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("Violin")
    n = len(c._qc_features_df)
    c._eda_launch = {"kind": "umap", "token": 1, "params": {},
                     "stack": c._current_stack_name, "n": n,
                     "plot": "Scatter"}
    c._on_eda_finished({"kind": "umap", "token": 1,
                        "projected": np.zeros((n, 2)), "input": "x",
                        "log": ""})
    assert wb.status_label.text() == T.MAP_DONE_PICK.format(map="UMAP",
                                                            input="x")


# ── LOW-6 ──

def test_a_second_click_after_a_mark_repicks_the_live_cells(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._toggle_category_selection("qc_tier", "good")
    pump()
    c._mark_bad()                       # the 5 frozen rows are bad now
    df = c._qc_features_df
    df.loc[df.label == 15, "qc_tier"] = "good"
    pw.refresh_marks(df)
    pw._toggle_category_selection("qc_tier", "good")   # the same entry
    pump()
    assert c._selected_label_ids == [15]      # re-picked, not toggled off
    pw._toggle_category_selection("qc_tier", "good")   # unchanged: off
    pump()
    assert c._selected_label_ids == []


def test_a_second_click_on_an_emptied_entry_clears(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._toggle_category_selection("qc_tier", "good")
    pump()
    c._mark_bad()
    pw._toggle_category_selection("qc_tier", "good")
    pump()
    assert c._selected_label_ids == []
    assert not pw._selected_categories

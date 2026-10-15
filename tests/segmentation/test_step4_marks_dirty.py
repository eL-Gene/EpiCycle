"""F8 / F18 / I-7: marks are unsaved work, keep the maps, survive Measure.

F8 and F18.  Qt-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.segmentation import _step4_text as T

from ._step4_fakes import controller


class _Plot:
    """The three map caches and a record of set_features calls."""

    def __init__(self):
        self._pca_cache = np.ones((3, 2))
        self._umap_cache = np.ones((3, 2))
        self._tsne_cache = np.ones((3, 2))
        self.calls = []

    def set_features(self, df, *, preserve_caches=False):
        self.calls.append(preserve_caches)
        if not preserve_caches:
            self._pca_cache = self._umap_cache = self._tsne_cache = None


class _AW:
    def __init__(self):
        self.qc_plot_widget = _Plot()
        self.fed = []

    def set_feature_data(self, *a, **k):
        self.fed.append(1)


@pytest.fixture
def ctrl(tmp_path):
    c = controller(tmp_path)
    c._current_stack_name = "s.tif"
    c._labels_volume = np.zeros((2, 2, 2), np.uint32)
    c._qc_features_df = pd.DataFrame({"label": [1, 2, 3]})
    c._analysis_widget = _AW()
    c.state_manager.set_stack_status("s.tif", "qc_extracted")
    c._open_analysis_dock = lambda: None
    return c


def test_a_mark_sets_tiers_dirty_and_keeps_the_maps(ctrl):
    ctrl._selected_label_ids = [1, 3]
    ctrl._mark_bad()
    assert ctrl._tiers_dirty is True                              # I-7
    assert ctrl._labels_dirty is False
    assert ctrl._has_unsaved_changes
    pw = ctrl._analysis_widget.qc_plot_widget
    assert pw.calls == [True]
    assert pw._pca_cache is not None and pw._umap_cache is not None
    assert pw._tsne_cache is not None
    assert list(ctrl._qc_features_df["qc_tier"]) == ["bad", "", "bad"]


def test_a_mark_never_greys_finalize(ctrl):
    ctrl._selected_label_ids = [2]
    ctrl._mark_good()
    ctrl._update_qc_button_states()
    assert ctrl.ui.finalize_btn.isEnabled()


def test_a_load_after_a_mark_prompts(ctrl, monkeypatch):
    """Unsaved marks ask Save / Discard / Stay; Stay keeps
    the stack and its marks."""
    import napari_epicycle.ui_shell.theme as theme
    asked = []
    monkeypatch.setattr(theme, "themed_three_choice",
                        lambda *a, **k: asked.append(a[1]) or "cancel")
    ctrl._selected_label_ids = [2]
    ctrl._mark_good()
    ctrl._request_load("other.tif")
    assert asked == [T.LOAD_UNSAVED_TITLE]
    assert ctrl._current_stack_name == "s.tif"
    assert ctrl._tiers_dirty


def test_measure_again_keeps_marks(ctrl):
    ctrl._qc_features_df = pd.DataFrame({"label": [1, 2, 3],
                                         "qc_tier": ["good", "", "bad"]})
    ctrl._measure_launch = {"token": 1, "stack": "s.tif",
                            "spacing": {"zyx": [2.0, 0.5, 0.5],
                                        "source": "user"},
                            "z_orientation": "basal_down"}
    new = pd.DataFrame({"label": [1, 3, 4], "area": [5, 6, 7]})
    ctrl._on_extract_finished(new, "s.tif")
    df = ctrl._qc_features_df
    assert list(df["qc_tier"]) == ["good", "bad", ""]      # 2 gone, 4 new
    assert ctrl._tiers_dirty
    assert any(T.MT_TIERS_KEPT.format(k=2) in s
               for s in ctrl.ui.qc_status_line.sets)
    rec = ctrl.state_manager.get_record("s.tif")
    assert rec["measured_spacing"] == {"zyx": [2.0, 0.5, 0.5],
                                       "source": "user"}
    assert rec["measured_z_orientation"] == "basal_down"
    assert rec["measured_cells"] == 3 and rec["measured_at"]


def test_measure_again_de_finalizes(ctrl):
    ctrl.state_manager.set_stack_status(
        "s.tif", "finalized", finalized_at="t",
        finalized_fingerprint={"labels": [1, 2], "csv": [3, 4]})
    ctrl._on_extract_finished(pd.DataFrame({"label": [1]}), "s.tif")
    rec = ctrl.state_manager.get_record("s.tif")
    assert rec["status"] == "qc_extracted"
    assert "finalized_at" not in rec and "finalized_fingerprint" not in rec


def test_remove_bad_cells_holds_its_edit_time(ctrl, monkeypatch):
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    vol = np.zeros((2, 3, 3), np.uint32)
    vol[0, 0, 0] = 1
    vol[1, 1, 1] = 2
    ctrl._labels_volume = vol
    ctrl._qc_features_df = pd.DataFrame({"label": [1, 2],
                                         "qc_tier": ["bad", "good"]})
    ctrl._delete_bad_labels()
    assert ctrl._labels_volume[0, 0, 0] == 0
    assert ctrl._labels_dirty
    assert ctrl._pending_edits["labels_changed_at"]
    assert "labels_changed_at" not in ctrl.state_manager.get_record("s.tif")


def test_measure_writes_only_the_saved_marks(ctrl, tmp_path, monkeypatch):
    """Re-audit M1: mark (unsaved), Measure -> the measured CSV carries
    the SAVED marks only, so Discard + reload shows no unsaved mark."""
    import napari_epicycle.segmentation.controller as C
    saved = tmp_path / "saved.csv"
    saved.write_text("label,qc_tier\n1,good\n2,\n3,\n", encoding="utf-8")
    ctrl.state_manager.update_record("s.tif", labels_path="L",
                                     qc_features_csv=str(saved))
    ctrl._qc_features_df = pd.DataFrame({"label": [1, 2, 3],
                                         "qc_tier": ["good", "", ""]})
    ctrl._selected_label_ids = [3]
    ctrl._mark_bad()                                  # unsaved
    got = {}

    class Spy:
        def __init__(self, **kw):
            got.update(kw)

    Real = C._StackQCFeatureWorker
    monkeypatch.setattr(C, "_StackQCFeatureWorker", Spy)
    ctrl._launch_measure_worker = lambda worker, token: object()
    ctrl.ui.chan1_combo = type("C", (), {"currentIndex": lambda s: 0})()
    ctrl._run_feature_extraction(skip_probability=True)
    assert not got["tier_map"]
    assert got["tier_csv"] == str(saved)
    worker = Real.__new__(Real)
    worker._tier_map, worker._tier_csv = {}, str(saved)
    worker.log = type("L", (), {"emit": lambda s, m: None})()
    assert Real._launch_tiers(worker) == {1: "good"}    # not label 3

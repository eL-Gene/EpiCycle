"""A clear that comes from the annotator must not echo back into it.

When the annotator drops a deleted cell from the isolation it tells the
controller, and the controller clears the plot.  The plot normally reports a
clear one tick later; that report would come back through the selection sync
and clear whatever the annotator isolated in between (the next row, with
auto-advance on) and untick the Isolate box.  A clear for a forgotten cell
therefore asks the plot not to report, and the controller does the few things
the report used to do.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from napari_epicycle.fusion_reconstruction import controller as ctl_mod
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


def _harness():
    h = MagicMock()
    h._colormap_authority = MagicMock()
    h._isolated_label_ids = {1}
    h._isolated_df_indices = [0]
    h._find_active_labels_layer = MagicMock(return_value=None)
    h.ui = MagicMock()
    h.ui.sync_viewer_colors_cb.isChecked.return_value = False
    h._selection_sync = SimpleNamespace(_last_key=("some", "key"))
    return h


def _call(h, **kw):
    FusionReconstructionController._on_annotator_isolation_cleared(h, **kw)


def test_a_forget_clear_asks_the_plot_not_to_report_and_does_the_echoed_work(monkeypatch):
    h = _harness()
    pushed = []
    monkeypatch.setattr(ctl_mod, "_area_call",
                        lambda c, name, *a, **k: pushed.append((name, a)))
    _call(h, reason="forget")
    h.ui.qc_plot_widget._clear_selection.assert_called_once_with(emit=False)
    assert h._selection_sync._last_key is None
    h._notify_plot_selection.assert_called_once()
    assert ("_plot_area_push", ("clear",)) in pushed


def test_the_default_clear_is_unchanged(monkeypatch):
    h = _harness()
    pushed = []
    monkeypatch.setattr(ctl_mod, "_area_call",
                        lambda c, name, *a, **k: pushed.append((name, a)))
    _call(h)
    h.ui.qc_plot_widget._clear_selection.assert_called_once_with()
    assert h._selection_sync._last_key == ("some", "key")
    h._notify_plot_selection.assert_not_called()
    assert pushed == []


def test_clearing_the_plot_selection_without_emit_queues_no_report(monkeypatch):
    from qtpy.QtCore import QTimer
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    shots = []
    monkeypatch.setattr(QTimer, "singleShot",
                        staticmethod(lambda ms, cb: shots.append(cb)))

    def _fake():
        return SimpleNamespace(
            _selected_indices=set(), _clicked_df_index=3,
            _selected_categories=set(), _selected_category_df_indices=[],
            _selection_lasso=MagicMock(), _lasso_points=[],
            _clear_sel_btn=MagicMock(), _stats_label=MagicMock(),
            _redraw=MagicMock(), labels_selected=MagicMock())

    quiet = _fake()
    QCPlotWidget._clear_selection(quiet, emit=False)
    assert shots == []
    loud = _fake()
    QCPlotWidget._clear_selection(loud)
    assert len(shots) == 1
    shots[0]()
    loud.labels_selected.emit.assert_called_once_with([])


def test_a_box_untick_clear_does_not_echo_into_the_navigator(monkeypatch):
    """Unticking Isolate must leave the table rows alone, so a re-tick isolates them again."""
    h = _harness()
    pushed = []
    monkeypatch.setattr(ctl_mod, "_area_call",
                        lambda c, name, *a, **k: pushed.append((name, a)))
    _call(h, reason="box_untick")
    h.ui.qc_plot_widget._clear_selection.assert_called_once_with(emit=False)
    assert h._selection_sync._last_key is None
    h._notify_plot_selection.assert_called_once()
    assert ("_plot_area_push", ("clear",)) in pushed


def test_the_other_reasons_still_report_the_clear(monkeypatch):
    """Merge Cancel, auto-advance and the like clear with a default reason: the
    plot reports it, so the row is released and can be clicked again."""
    for reason in ("toggle_off", None):
        h = _harness()
        monkeypatch.setattr(ctl_mod, "_area_call", lambda *a, **k: None)
        if reason is None:
            _call(h)
        else:
            _call(h, reason=reason)
        h.ui.qc_plot_widget._clear_selection.assert_called_once_with()
        assert h._selection_sync._last_key == ("some", "key")

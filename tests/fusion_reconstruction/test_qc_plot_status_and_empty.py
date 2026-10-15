"""``set_status``, the instance
``status_changed``, ``set_empty_sentence``, the workbench ``clear()`` and
the kept-selection hook.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fusion_reconstruction import _qc_plot_snap as S

GOLDEN = Path(__file__).with_name("golden") / "qc_plot_workbench.json"


def _wb(**kw):
    S.app()
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME,
        QCPlotWidget,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    w = QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS,
                     palette=STEP3_PLOT_PALETTE, tier_scheme=STEP3_SCHEME,
                     **kw)
    w.resize(562, 600)
    return w


@pytest.fixture
def wb():
    w = _wb()
    yield w
    w.deleteLater()


def _axes_visible(w):
    item = w._plot_widget.getPlotItem()
    return (item.getAxis("bottom").isVisible(),
            item.getAxis("left").isVisible())


# --- set_status ----------------------------------------------------------------


def test_set_status_emits_under_the_workbench(wb):
    seen = []
    wb.status_changed.connect(lambda t, k: seen.append((t, k)))
    wb.set_status("Assigned 15 labels to review.")
    wb.set_status("Isolation failed — toggle Isolate off and retry.",
                  "error")
    wb.set_status("Careful.", kind="warn")
    assert seen == [("Assigned 15 labels to review.", "info"),
                    ("Isolation failed — toggle Isolate off and retry.",
                     "error"),
                    ("Careful.", "warn")]
    assert wb.status() == ("Careful.", "warn")
    # the plot's own (hidden) count is not written by a status
    assert "Careful" not in wb._stats_label.text()
    with pytest.raises(ValueError):
        wb.set_status("x", "ok")


def test_set_status_writes_the_label_under_classic():
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    w = QCPlotWidget()
    try:
        w.set_status("Viewer: 97 labels shown alone")
        assert w._stats_label.text() == "Viewer: 97 labels shown alone"
        assert not hasattr(w, "status_changed")
        with pytest.raises(RuntimeError):
            w.status()
    finally:
        w.deleteLater()


def test_no_new_class_signal():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))["class_signals"]
    got = sorted(k for k, v in vars(QCPlotWidget).items()
                 if type(v).__name__ in ("pyqtSignal", "Signal"))
    assert got == expected


def test_status_changed_is_an_instance_signal_with_no_widget(wb):
    from qtpy.QtWidgets import QWidget

    assert "status_changed" in vars(wb)
    assert "status_changed" not in vars(type(wb))
    holder = wb._wb_status_notifier
    assert not isinstance(holder, QWidget)
    assert holder.parent() is wb


# --- set_empty_sentence --------------------------------------------------------


def test_the_empty_sentence_is_drawn_and_cleared(wb):
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        PLOT_NO_FEATURES,
        PLOT_NO_PATCH,
    )

    assert S.plot_texts(wb) == []
    wb.set_empty_sentence(PLOT_NO_PATCH)
    assert S.plot_texts(wb) == [PLOT_NO_PATCH.replace("|", "\n")]
    assert _axes_visible(wb) == (False, False)
    wb.set_empty_sentence(PLOT_NO_FEATURES)
    assert S.plot_texts(wb) == [PLOT_NO_FEATURES.replace("|", "\n")]
    wb.set_empty_sentence(None)
    assert S.plot_texts(wb) == []
    assert _axes_visible(wb) == (True, True)


def test_the_empty_sentence_shows_on_a_table_with_no_rows(wb):
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        PLOT_NO_FEATURES,
    )

    wb.set_empty_sentence(PLOT_NO_FEATURES)
    wb.set_features(S.five_tier_df().iloc[0:0])
    assert S.plot_texts(wb) == [PLOT_NO_FEATURES.replace("|", "\n")]


def test_a_loaded_plot_never_shows_the_sentence(wb):
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        PLOT_NO_PATCH,
    )

    wb.set_features(S.five_tier_df())
    wb.set_empty_sentence(PLOT_NO_PATCH)
    assert S.plot_texts(wb) == []
    assert _axes_visible(wb) == (True, True)


def test_classic_has_no_empty_sentence():
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    w = QCPlotWidget()
    try:
        with pytest.raises(RuntimeError):
            w.set_empty_sentence("x|y")
    finally:
        w.deleteLater()


# --- clear() ------------------------------------------------------------------


def test_workbench_clear_hides_the_gradient_greys_clear_and_draws(wb):
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        PLOT_NO_PATCH,
    )

    wb.set_empty_sentence(PLOT_NO_PATCH)
    wb.set_features(S.five_tier_df())
    assert wb.set_colour_by("area")
    assert not wb._wb_gradient.isHidden()
    wb.highlight_labels([1, 2, 3])
    assert wb._wb_clear_btn.isEnabled()
    wb.clear()
    assert wb._wb_gradient.isHidden()
    assert not wb._legend_label.isHidden()
    assert not wb._wb_clear_btn.isEnabled()
    assert S.plot_texts(wb) == [PLOT_NO_PATCH.replace("|", "\n")]
    assert wb.selected_label_ids() == []


def test_workbench_clear_without_a_sentence_draws_nothing(wb):
    wb.set_features(S.five_tier_df())
    wb.clear()
    assert S.plot_texts(wb) == []
    assert not wb._wb_clear_btn.isEnabled()


# --- the kept-selection hook ------------------------------------------------


def test_the_kept_hook_runs_when_a_switch_keeps_the_selection(wb):
    calls = []
    wb.set_selection_kept_callback(lambda: calls.append(
        wb.selected_label_ids()))
    wb.set_features(S.five_tier_df())
    wb.plot_type_combo.setCurrentText("PCA")       # nothing selected
    assert calls == []
    wb._toggle_category_selection("qc_tier", "bad")
    picked = wb.selected_label_ids()
    wb.plot_type_combo.setCurrentText("Scatter")
    assert calls == [picked]
    wb.set_selection_kept_callback(None)
    wb.plot_type_combo.setCurrentText("PCA")
    assert len(calls) == 1


def test_a_failing_kept_hook_never_breaks_the_switch(wb, capsys):
    def boom():
        raise RuntimeError("host bug")

    wb.set_selection_kept_callback(boom)
    wb.set_features(S.five_tier_df())
    wb._toggle_category_selection("qc_tier", "good")
    wb.plot_type_combo.setCurrentText("PCA")
    assert wb.plot_type_combo.currentText() == "PCA"
    assert wb._wb_type_buttons["PCA"].isChecked()
    assert "[QC-PLOT] kept-selection callback failed" in capsys.readouterr().out
    with pytest.raises(TypeError):
        wb.set_selection_kept_callback("not callable")


def test_the_kept_hook_is_workbench_only():
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    w = QCPlotWidget()
    try:
        with pytest.raises(RuntimeError):
            w.set_selection_kept_callback(lambda: None)
    finally:
        w.deleteLater()


def _sentence_item(w):
    import pyqtgraph as pg

    items = [it for it in w._plot_widget.getPlotItem().items
             if isinstance(it, pg.TextItem)]
    assert len(items) == 1
    return items[0]


def test_step3_sentence_is_centred_in_the_application_font(wb):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        PLOT_NO_PATCH,
    )

    wb.set_empty_sentence(PLOT_NO_PATCH)
    it = _sentence_item(wb)
    assert tuple(it.anchor) == (0.5, 0.5)
    opt = it.textItem.document().defaultTextOption()
    assert opt.alignment() & Qt.AlignHCenter
    assert it.textItem.font().family() == QApplication.font().family()
    # the empty map shares the drawing
    wb.set_empty_sentence(None)
    wb.set_features(S.five_tier_df())
    wb.plot_type_combo.setCurrentText("UMAP")
    assert _sentence_item(wb).textItem.document().defaultTextOption(
    ).alignment() & Qt.AlignHCenter


def test_step4_empty_map_is_drawn_as_before():
    """Step 4 passes no palette: its empty map keeps the left-aligned
    10 pt sans-serif text it always had."""
    from qtpy.QtCore import Qt

    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_PLOT_TEXTS,
    )

    w = QCPlotWidget(layout="workbench", texts=WORKBENCH_PLOT_TEXTS)
    try:
        w.set_features(S.five_tier_df())
        w.plot_type_combo.setCurrentText("UMAP")
        it = _sentence_item(w)
        assert tuple(it.anchor) == (0.5, 0.5)
        f = it.textItem.font()
        assert (f.family(), f.pointSize()) == ("sans-serif", 10)
        assert not (it.textItem.document().defaultTextOption().alignment()
                    & Qt.AlignHCenter)
    finally:
        w.deleteLater()


def test_status_starts_empty_and_an_empty_status_clears(wb):
    assert wb.status() == ("", "info")
    seen = []
    wb.status_changed.connect(lambda t, k: seen.append((t, k)))
    wb.set_status("Undid the last assignment.")
    wb.set_status("")
    assert wb.status() == ("", "info")
    wb.set_status(None, "warn")
    assert seen == [("Undid the last assignment.", "info"), ("", "info"),
                    ("", "warn")]


def test_umap_with_no_table_shows_the_host_sentence_not_the_map_one(wb):
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        PLOT_NO_PATCH,
        UMAP_EMPTY,
    )

    wb.plot_type_combo.setCurrentText("UMAP")
    assert S.plot_texts(wb) == []
    wb.set_empty_sentence(PLOT_NO_PATCH)
    assert S.plot_texts(wb) == [PLOT_NO_PATCH.replace("|", "\n")]
    wb.set_features(S.five_tier_df())          # a table, but no UMAP yet
    assert S.plot_texts(wb) == [UMAP_EMPTY.replace("|", "\n")]
    assert "3,000 labels" in S.plot_texts(wb)[0]

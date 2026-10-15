"""The Predict + Reconstruct section on Step 4's ② layout, in amber
(brick R3).

The run row full width with the lock line's slot under it; the splitter
240 | 330 (6 px handle, minimums 240 | 300, not collapsible, stretch 0 : 1);
the Model and Patches cards on the left; R2a's settings column in a scroll
area over the PINNED foot on the right; ONE status setter with three kinds;
Session's Model Status hidden behind its pointer line; and the foot's gate
table on the real controller.

Most tests build the UI WITHOUT a controller (``__new__`` + ``_build_ui``),
as the parity tests do; the gate tests build the whole widget.  Heights
that depend on the font live in ``test_step3_pr_layout_ceilings.py``.  Run
one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from tests.fusion_reconstruction._platform import windows_font_metrics

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

FR = Path(__file__).resolve().parents[2] / "napari_epicycle" / \
    "fusion_reconstruction"


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _pump(app, n=5):
    for _ in range(n):
        app.processEvents()


def _bare():
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    return w


@pytest.fixture(scope="module")
def ui(app):
    w = _bare()
    yield w
    w.deleteLater()


def _P():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    return P


# ---------------------------------------------------------------- the frame

def test_the_splitter_has_step_4s_numbers(app, ui):
    from qtpy.QtCore import Qt
    sp = ui.pr_splitter
    assert sp.orientation() == Qt.Horizontal
    assert not sp.childrenCollapsible()
    assert sp.handleWidth() == 6
    assert sp.widget(0) is ui.pr_left and sp.widget(1) is ui.pr_right
    assert ui.pr_left.minimumWidth() == 240
    assert ui.pr_right.minimumWidth() == 300
    assert ui.pr_left.layout().contentsMargins().right() == 6
    assert ui.pr_left.layout().contentsMargins().left() == 0
    assert ui.pr_right.layout().contentsMargins().left() == 8


def test_the_left_column_stays_at_240_as_the_dock_widens(app):
    w = _bare()
    try:
        w.resize(700, 1400)
        w.show()
        _pump(app, 10)
        assert w.pr_splitter.sizes()[0] == 240
        right = w.pr_splitter.sizes()[1]
        w.resize(1200, 1400)
        _pump(app, 10)
        assert w.pr_splitter.sizes()[0] == 240      # stretch 0 : 1
        assert w.pr_splitter.sizes()[1] > right
    finally:
        w.close()
        w.deleteLater()


def test_the_foot_is_pinned_outside_the_scroll_area(ui):
    from qtpy.QtCore import Qt
    assert ui.pr_foot.parent() is ui.pr_right
    assert ui.pr_right_scroll.widget() is ui.pr_right_content
    for b in (ui.predict_selected_btn, ui.reconstruct_selected_btn,
              ui.pipeline_cancel_btn, ui.view_reconstructed_button,
              ui.pipeline_progress, ui.pipeline_status_label):
        assert ui.pr_foot.isAncestorOf(b)
        assert not ui.pr_right_content.isAncestorOf(b)
    assert ui.pr_right_content.isAncestorOf(ui._pr_settings)
    assert ui.pr_right_scroll.horizontalScrollBarPolicy() == \
        Qt.ScrollBarAlwaysOff


def test_the_foot_does_not_move_when_the_settings_scroll(app):
    w = _bare()
    try:
        w.resize(700, 900)
        w.show()
        w._pr_adv_collapsible.expand(animate=False)
        for g in w._pr_adv_groups.values():
            g.expand(animate=False)
        _pump(app, 10)
        bar = w.pr_right_scroll.verticalScrollBar()
        assert bar.maximum() > 0          # the settings DO scroll here
        panel = w._pr_panel
        foot_y = w.pr_foot.mapTo(panel, w.pr_foot.rect().topLeft()).y()
        btn_y = w.predict_selected_btn.mapTo(
            panel, w.predict_selected_btn.rect().topLeft()).y()
        start_y = w.start_from_combo.mapTo(
            panel, w.start_from_combo.rect().topLeft()).y()
        bar.setValue(bar.maximum())
        _pump(app, 10)
        assert w.pr_foot.mapTo(
            panel, w.pr_foot.rect().topLeft()).y() == foot_y
        assert w.predict_selected_btn.mapTo(
            panel, w.predict_selected_btn.rect().topLeft()).y() == btn_y
        # ...while the settings themselves moved
        assert w.start_from_combo.mapTo(
            panel, w.start_from_combo.rect().topLeft()).y() < start_y
    finally:
        w.close()
        w.deleteLater()


def test_the_foot_is_two_by_two(ui):
    grid = ui.predict_selected_btn.parentWidget().layout().itemAt(0).layout()
    pos = {}
    for b in (ui.predict_selected_btn, ui.reconstruct_selected_btn,
              ui.pipeline_cancel_btn, ui.view_reconstructed_button):
        idx = grid.indexOf(b)
        assert idx >= 0
        pos[b] = grid.getItemPosition(idx)[:2]
    assert pos[ui.predict_selected_btn] == (0, 0)
    assert pos[ui.reconstruct_selected_btn] == (0, 1)
    assert pos[ui.pipeline_cancel_btn] == (1, 0)
    assert pos[ui.view_reconstructed_button] == (1, 1)


def test_the_same_button_objects_and_every_alias(ui):
    """``_CURATION_LOCK_WIDGETS`` and the controller's old names still point
    at the foot's widgets."""
    assert ui.segment_button is ui.predict_selected_btn
    assert ui.batch_process_btn is ui.predict_selected_btn
    assert ui.run_reconstruction_btn is ui.reconstruct_selected_btn
    for name in ("prediction_pause_btn", "reconstruction_pause_btn",
                 "batch_pause_btn"):
        assert getattr(ui, name) is ui.pipeline_cancel_btn
    for name in ("prediction_progress", "reconstruction_progress"):
        assert getattr(ui, name) is ui.pipeline_progress
    for name in ("prediction_status_label", "reconstruction_status_label",
                 "batch_progress_label"):
        assert getattr(ui, name) is ui.pipeline_status_label
    for name in ("_predicted_bucket", "_reconstructed_bucket",
                 "reconstructed_results_list"):
        assert getattr(ui, name) is ui._patch_state_list
    assert ui._recon_column is ui.pr_right
    assert ui.processing_mode_combo.isHidden()
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    for name in ("predict_selected_btn", "reconstruct_selected_btn"):
        assert name in C._CURATION_LOCK_WIDGETS
        assert ui.pr_foot.isAncestorOf(getattr(ui, name))


def test_the_run_row_is_full_width_with_the_lock_line_over_it(ui):
    """V1: the lock line sits ABOVE the run row (the mockup's board F,
    Step 4's ②), hidden while no flag is set."""
    lay = ui._pr_panel.layout()
    assert lay.itemAt(0).widget() is ui.pr_lock_line
    run_row = lay.itemAt(1).layout()
    widgets = [run_row.itemAt(i).widget() for i in range(run_row.count())]
    assert ui.run_picker_combo in widgets
    assert ui.new_run_button in widgets
    assert ui.manage_runs_button in widgets
    assert widgets[0].text() == _P().S3_RUN_LABEL
    assert ui.pr_lock_line.isHidden()
    assert ui.pr_lock_line.objectName() == "step3LockLine"
    assert not ui.pr_splitter.isAncestorOf(ui.run_picker_combo)


# ------------------------------------------------------------ the two cards

def test_the_model_card(app, ui):
    from qtpy.QtWidgets import QLabel
    P = _P()
    assert ui.pr_left.isAncestorOf(ui.model_change_btn)
    assert ui.model_change_btn.text() == P.S3_MODEL_CHANGE_BTN
    texts = [lb.text() for lb in ui.pr_left.findChildren(QLabel)]
    assert P.S3_MODEL_DESC in texts
    assert P.S3_MODEL_ACTIVE_LABEL in texts
    assert P.S3_MODEL_TITLE in texts
    ui.show_pr_model("wound-v2", "", "C:/m/wound-v2/lateral", "")
    assert ui.pr_model_lat_label.text() == "lat: wound-v2"
    assert ui.pr_model_ax_label.text() == "ax: " + P.S3_MODEL_NO_NAME
    assert "C:/m/wound-v2/lateral" in ui.pr_model_lat_label.toolTip()
    # neutral, never amber
    for lbl in (ui.pr_model_lat_label, ui.pr_model_ax_label):
        assert "#cccccc" in lbl.styleSheet()
        assert "#d4a843" not in lbl.styleSheet()
        assert lbl.minimumSizeHint().width() == 0     # elided, never wide


def test_a_long_model_name_is_elided_in_the_middle(app):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QFontMetrics
    w = _bare()
    try:
        w.resize(700, 1200)
        w.show()
        name = "wound-v2-membrane-nuclei-refined-2026-10-04 · round 12"
        w.show_pr_model(name, name, name, name)
        _pump(app, 5)
        lbl = w.pr_model_lat_label
        assert lbl.text() == "lat: " + name                 # text stays whole
        fm = QFontMetrics(lbl.font())
        shown = fm.elidedText(lbl.text(), Qt.ElideMiddle,
                               lbl.contentsRect().width())
        assert shown.startswith("lat: ") and shown.endswith("round 12")
    finally:
        w.close()
        w.deleteLater()


def test_the_hidden_model_state_stays_built_and_hidden(ui):
    for name in ("lateral_model_path", "axial_model_path",
                 "pretrained_combo", "lateral_model_browse",
                 "axial_model_browse", "model_source_combo",
                 "model_active_indicator_label"):
        w = getattr(ui, name)
        assert ui._hidden_model_state.isAncestorOf(w), name
    assert ui._hidden_model_state.isHidden()
    assert ui.pr_left.isAncestorOf(ui._hidden_model_state)


def test_the_patches_card(ui):
    from qtpy.QtWidgets import QLabel, QListWidget
    P = _P()
    texts = [lb.text() for lb in ui.pr_left.findChildren(QLabel)]
    assert P.S3_PATCHES_TITLE in texts
    assert P.S3_PATCHES_DESC in texts
    assert ui.pr_left.isAncestorOf(ui._patch_state_list)
    # V1 (row 20): the ticks choose; ONE highlighted row is View's
    assert ui._patch_state_list.selectionMode() == \
        QListWidget.SingleSelection
    assert ui.pr_tick_all_link.text() == P.S3_TICK_ALL_LINK
    assert "underline" in ui.pr_tick_all_link.styleSheet()
    ui.set_pr_ticked_lines(4, 1, 3)
    assert ui.pr_ticked_predict_label.text() == "Predict: 4 ticked · 1 protected"
    assert ui.pr_ticked_recon_label.text() == "Reconstruct: 3 ready of 4 ticked"
    ui.set_pr_ticked_lines(2, 0, 0)
    assert ui.pr_ticked_predict_label.text() == "Predict: 2 ticked"


def test_the_two_cards_open_their_guides(ui):
    guides = [g for _b, g in ui._pr_help_buttons]
    assert guides == ["step3_predict_reconstruct_model_source.md",
                      "step3_predict_reconstruct_patch_state.md"]
    root = Path(__file__).resolve().parents[2] / "docs" / "guides"
    for g in guides:
        assert (root / g).exists()


def test_every_layout_help_line_is_a_tooltip(ui):
    for name, line in _P().S3_LAYOUT_HELP_LINES.items():
        assert getattr(ui, name).toolTip() == line, name


def test_the_patch_list_has_an_honest_help_line(ui):
    """R3 audit LOW-4, then V1: the line says the ticks are the choice."""
    line = _P().S3_LAYOUT_HELP_LINES.get("_patch_state_list", "")
    assert line == _P().S3_LIST_HELP
    assert "Ctrl+A" in line and "Tick" in line
    assert ui._patch_state_list.toolTip() == line


def test_ticks_are_gold_over_the_whole_section(ui):
    from napari_epicycle.ui_shell.theme import PANEL_CHECKBOX_STYLE
    assert ui._pr_panel.styleSheet() == PANEL_CHECKBOX_STYLE
    assert ui._pr_settings.styleSheet() == PANEL_CHECKBOX_STYLE


def test_no_old_header_or_blue_header_is_left(ui):
    from qtpy.QtWidgets import QLabel
    texts = [lb.text() for lb in ui._pr_panel.findChildren(QLabel)]
    for old in ("Model Source", "Patch State", "Current patches to process.",
                "Select a model to use for prediction."):
        assert old not in texts, old
    for lb in ui._pr_panel.findChildren(QLabel):
        assert "#42a5f5" not in lb.styleSheet()


def test_every_section_widget_has_a_parent(ui):
    from qtpy.QtWidgets import QWidget
    for name in ("run_picker_combo", "new_run_button", "manage_runs_button",
                 "pr_lock_line", "pr_splitter", "pr_left", "pr_right",
                 "pr_right_scroll", "pr_right_content", "pr_foot",
                 "model_change_btn", "pr_model_lat_label",
                 "pr_model_ax_label", "_patch_state_list",
                 "pr_ticked_predict_label", "pr_ticked_recon_label",
                 "pr_tick_all_link", "predict_selected_btn",
                 "reconstruct_selected_btn", "pipeline_cancel_btn",
                 "view_reconstructed_button", "pipeline_progress",
                 "pipeline_status_label", "processing_mode_combo"):
        w = getattr(ui, name)
        assert isinstance(w, QWidget)
        assert ui._pr_panel.isAncestorOf(w), name


# ----------------------------------------------------------- the status line

def test_show_pr_status_has_three_kinds(ui):
    from napari_epicycle.ui_shell.theme import (
        MONO_LABEL_STYLE, PANEL_ERROR_STYLE, PANEL_WARN_STYLE,
    )
    lbl = ui.pipeline_status_label
    ui.show_pr_status("Ready", "neutral")
    assert lbl.styleSheet() == MONO_LABEL_STYLE
    ui.show_pr_status("Some failed", "warn")
    assert lbl.styleSheet() == PANEL_WARN_STYLE
    ui.show_pr_status("All failed", "error")
    assert lbl.styleSheet() == PANEL_ERROR_STYLE
    ui.show_pr_status("Hm", "nonsense")
    assert lbl.styleSheet() == MONO_LABEL_STYLE
    ui.show_pr_status(None)
    assert lbl.text() == ""


def test_the_status_is_two_lines_at_most_and_the_tooltip_whole(app):
    w = _bare()
    try:
        w.resize(566, 1200)
        w.show()
        _pump(app, 5)
        long = " ".join(["Segmenting patch 12/40: Wound_patch_3.tiff"] * 8)
        w.show_pr_status(long, "warn")
        lbl = w.pipeline_status_label
        assert lbl.toolTip() == long
        assert lbl.text().count("\n") == 1
        assert lbl.text().endswith("…")
        assert lbl.maximumHeight() < 4 * lbl.fontMetrics().lineSpacing()
    finally:
        w.close()
        w.deleteLater()


def _longest_run_sentence():
    from napari_epicycle.ui_shell import theme as T
    return T.PIPE_SEGMENTING_VIEW.format(
        unit="patch", i=12, n=40,
        name="WholeScale_Wound_patch_3_processed.tif", VIEW="XZ", c=123,
        t=480, time=T.pipe_time("1:23:45", "2:10:00"))


@pytest.mark.parametrize("kind", [
    "neutral",
    pytest.param("warn", marks=windows_font_metrics),
    pytest.param("error", marks=windows_font_metrics),
])
def test_the_boxed_status_never_clips_its_second_line(app, kind):
    """R3 audit MED-1: the warn / error box (padding 5/7 + a 1 px border)
    narrows the label's CONTENTS; the two lines are fitted at the contents
    width after the style is applied, so the wrapped text never needs more
    height than the label is allowed (at the 566 px panel the label is
    300 px wide, after the 240 | 330 column change of A3)."""
    w = _bare()
    try:
        w.resize(566, 1200)
        w.show()
        _pump(app, 5)
        lbl = w.pipeline_status_label
        assert lbl.width() == 300      # A3: 566 - 240 - 6 - 8 - 12
        w.show_pr_status(_longest_run_sentence(), kind)
        _pump(app, 3)
        assert lbl.text().count("\n") <= 1
        assert lbl.heightForWidth(lbl.width()) <= lbl.maximumHeight(), (
            lbl.text(), lbl.contentsRect().width())
    finally:
        w.close()
        w.deleteLater()


def test_the_first_line_does_not_break_inside_the_time_bracket(app):
    """The time part ``[1:23:45 | 2:10:00]`` is kept whole on the first
    line: never "[1:23:45 |" there and "2:10:00]" below."""
    from napari_epicycle.fusion_reconstruction.ui import _two_lines
    w = _bare()
    try:
        w.resize(566, 1200)
        w.show()
        _pump(app, 5)
        lbl = w.pipeline_status_label
        fm = lbl.fontMetrics()
        head = "Reconstructing 3/40: Wound_patch_3.tif..."
        # a width that ends INSIDE the bracket
        cut = fm.horizontalAdvance(head + " [1:23:45 |") + 2
        assert cut > 200
        lbl.setFixedWidth(cut)
        text = (head + " " + " ".join(["[1:23:45", "|", "2:10:00]"])
                + " and a tail that will not fit")
        first = _two_lines(lbl, text).split("\n")[0]
        assert first.count("[") == first.count("]"), first
    finally:
        w.close()
        w.deleteLater()


# ------------------------------------------------------------------ Session

def test_session_hides_model_status_behind_its_pointer(ui):
    from napari_epicycle.ui_shell.theme import DIM_LABEL_STYLE
    box = ui._session_model_status
    assert box.isHidden()
    assert box.isAncestorOf(ui.status_lateral_model_value)
    assert box.isAncestorOf(ui.status_axial_model_value)
    ptr = ui.session_model_pointer
    assert ptr.text() == _P().S3_SESSION_MODEL_POINTER
    assert ptr.styleSheet() == DIM_LABEL_STYLE
    assert not ptr.isHidden()
    # A2 (board B): under the Pipeline status block, which follows the
    # hidden block
    lay = box.parentWidget().layout()
    assert lay.indexOf(ui._session_dashboard) == lay.indexOf(box) + 1
    assert lay.indexOf(ptr) == lay.indexOf(ui._session_dashboard) + 1


# ------------------------------------------------------- the gates (real)

@pytest.fixture
def full(app):
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration(viewer=None)
    c = w.controller
    statuses = {}
    c.state_manager.get_all_patch_statuses = lambda: dict(statuses)
    yield w, c, statuses
    w.deleteLater()


def _list(w, c, statuses, rows):
    statuses.clear()
    statuses.update(rows)
    w.refresh_patch_state_list(dict(rows))


def _select(w, *names):
    """V1 (row 20): TICK exactly *names* (the ticks choose what runs; the
    user's tick goes through ``itemChanged``)."""
    from qtpy.QtCore import Qt
    lst = w._patch_state_list
    for i in range(lst.count()):
        item = lst.item(i)
        want = Qt.Checked if item.data(Qt.UserRole) in names \
            else Qt.Unchecked
        if item.checkState() != want:
            item.setCheckState(want)


def _status(w) -> str:
    """The whole sentence (the label itself may carry a wrap the two-line
    cut inserted; its tooltip is the sentence)."""
    return w.pipeline_status_label.toolTip()


def _models(w, c, on=True):
    w.lateral_model_path.setText("cyto3" if on else "")
    w.axial_model_path.setText("cyto3" if on else "")
    c._refresh_model_status()


def test_no_model_says_pick_a_model(full):
    w, c, statuses = full
    P = _P()
    _list(w, c, statuses, {"a.tif": "unprocessed"})
    _models(w, c, on=False)
    _select(w, "a.tif")
    assert not w.predict_selected_btn.isEnabled()
    assert _status(w) == P.S3_GATE_PICK_MODEL
    assert w.pr_model_lat_label.text() == "lat: " + P.S3_MODEL_NO_NAME


def test_nothing_selected_greys_both(full):
    w, c, statuses = full
    P = _P()
    _list(w, c, statuses, {"a.tif": "unprocessed", "b.tif": "predicted"})
    _models(w, c)
    _select(w)
    assert not w.predict_selected_btn.isEnabled()
    assert not w.reconstruct_selected_btn.isEnabled()
    assert _status(w) == P.S3_GATE_NOTHING_TICKED
    assert w.pr_ticked_predict_label.text() == "Predict: 0 ticked"


def test_an_unprocessed_patch_can_predict_not_reconstruct(full):
    w, c, statuses = full
    P = _P()
    _list(w, c, statuses, {"a.tif": "unprocessed", "b.tif": "predicted"})
    _models(w, c)
    _select(w, "a.tif")
    assert w.predict_selected_btn.isEnabled()
    assert not w.reconstruct_selected_btn.isEnabled()
    assert _status(w) == (
        "Ready · 1 ticked" + P.S3_GATE_JOIN
        + P.S3_GATE_NOTHING_TO_RECONSTRUCT)
    assert w.pr_ticked_recon_label.text() == \
        "Reconstruct: 0 ready of 1 ticked"


def test_a_predicted_patch_lights_both(full):
    w, c, statuses = full
    _list(w, c, statuses, {"a.tif": "unprocessed", "b.tif": "predicted"})
    _models(w, c)
    _select(w, "a.tif", "b.tif")
    assert w.predict_selected_btn.isEnabled()
    assert w.reconstruct_selected_btn.isEnabled()
    assert _status(w) == "Ready · 2 ticked"
    assert w.pr_ticked_recon_label.text() == \
        "Reconstruct: 1 ready of 2 ticked"


def test_the_gates_have_the_last_word_after_the_bulk_unlock(full):
    """``_refresh_step3_lock`` lights both buttons on unlock; the gates run
    after it, so nothing selected keeps them grey (Step 4's R1-1 lesson:
    the control is first proved live without the reason)."""
    w, c, statuses = full
    _list(w, c, statuses, {"b.tif": "predicted"})
    _models(w, c)
    _select(w, "b.tif")
    assert w.predict_selected_btn.isEnabled()        # live without a reason
    _select(w)
    c._refresh_step3_lock()
    assert not w.predict_selected_btn.isEnabled()
    assert not w.reconstruct_selected_btn.isEnabled()


def test_a_running_job_keeps_its_greys_and_its_line(full):
    w, c, statuses = full
    _list(w, c, statuses, {"b.tif": "predicted"})
    _models(w, c)
    _select(w, "b.tif")
    c._prediction_running = True
    c._refresh_step3_lock()
    c._pr_status("Segmenting 1 patches (10 slices)...")
    c._refresh_pr_foot_gates(keep_status=False)
    assert not w.predict_selected_btn.isEnabled()
    assert not w.reconstruct_selected_btn.isEnabled()
    assert _status(w) == \
        "Segmenting 1 patches (10 slices)..."
    c._prediction_running = False
    c._refresh_step3_lock()


def test_a_non_run_lock_blanks_the_status(full):
    """R3 audit LOW-1 (Step 4's rule): under a lock that is not Predict /
    Reconstruct the buttons are grey, so the foot never reads "Ready"
    beside them -- the status is blank; on unlock the gates speak again."""
    w, c, statuses = full
    _list(w, c, statuses, {"a.tif": "unprocessed"})
    _models(w, c)
    _select(w, "a.tif")
    assert _status(w).startswith("Ready")          # live without the lock
    c._curation_busy = True
    try:
        c._refresh_step3_lock()
        c._refresh_pr_foot_gates()
        assert not w.predict_selected_btn.isEnabled()
        assert _status(w) == ""
        assert w.pipeline_status_label.text() == ""
    finally:
        c._curation_busy = False
    c._refresh_step3_lock()
    assert _status(w).startswith("Ready")


def test_a_session_click_on_an_unpredicted_patch_lets_the_gates_speak(
        full, monkeypatch):
    """R3 audit LOW-2: a Session-list click on a patch with no prediction
    wrote "" on the foot, wiping the gate line; it now re-gates."""
    w, c, statuses = full
    _list(w, c, statuses, {"a.tif": "unprocessed"})
    _models(w, c)
    _select(w, "a.tif")
    c._pr_status("Segmentation complete: 1/1 patches  [5s]")
    monkeypatch.setattr(c, "_selected_patch_name", lambda: "a.tif")
    monkeypatch.setattr(c.state_manager, "get_patch_status",
                        lambda name: "unprocessed")
    monkeypatch.setattr(c, "_reload_current_patch_features", lambda: None)
    c._on_patch_selected()
    assert _status(w).startswith("Ready")


def test_a_session_click_says_restored_or_no_cache_in_step3s_words(
        full, monkeypatch):
    P = _P()
    w, c, statuses = full
    monkeypatch.setattr(c, "_selected_patch_name", lambda: "b.tif")
    monkeypatch.setattr(c.state_manager, "get_patch_status",
                        lambda name: "predicted")
    monkeypatch.setattr(c, "_reload_current_patch_features", lambda: None)
    c.loaded_stack_dir = "D:/ws/patches"
    monkeypatch.setattr(c, "_prediction_cache_exists", lambda name: True)
    c._on_patch_selected()
    assert _status(w) == P.S3_STATUS_PREDICTIONS_CACHED.format(
        patch="b.tif")
    monkeypatch.setattr(c, "_prediction_cache_exists", lambda name: False)
    c._on_patch_selected()
    assert _status(w) == P.S3_STATUS_NO_CACHE.format(status="predicted")


def test_the_gates_print_only_under_the_debug_gate(full, monkeypatch,
                                                  capsys):
    """R3 audit LOW-3: ``[STEP3-PR] gates ...`` is a dev print, gated by
    ``EPICYCLE_SYNC_DEBUG=1`` like ``[BATCH-PERF]``."""
    w, c, statuses = full
    _list(w, c, statuses, {"a.tif": "unprocessed"})
    _models(w, c)
    _select(w, "a.tif")
    monkeypatch.delenv("EPICYCLE_SYNC_DEBUG", raising=False)
    capsys.readouterr()
    c._refresh_pr_foot_gates(keep_status=False)
    assert "[STEP3-PR] gates" not in capsys.readouterr().out
    monkeypatch.setenv("EPICYCLE_SYNC_DEBUG", "1")
    c._refresh_pr_foot_gates(keep_status=False)
    assert "[STEP3-PR] gates" in capsys.readouterr().out


def test_a_finished_line_stays_until_the_ticks_change(full):
    w, c, statuses = full
    _list(w, c, statuses, {"a.tif": "unprocessed", "b.tif": "predicted"})
    _models(w, c)
    _select(w, "b.tif")
    c._pr_status("Segmentation complete: 1/1 patches  [5s]", "warn")
    c._refresh_step3_lock()                       # keep_status
    assert _status(w).startswith("Segmentation complete")
    _select(w, "a.tif")                           # a tick change
    assert _status(w).startswith("Ready")


def test_a_list_rebuild_is_not_a_selection_change(full):
    """``refresh_patch_state_list`` clears the list; that must not count as
    the user changing the patches (it would wipe a job's line)."""
    w, c, statuses = full
    _list(w, c, statuses, {"b.tif": "predicted"})
    _models(w, c)
    _select(w, "b.tif")
    c._pr_status("Reconstruction complete: 1/1 patches  [5s]")
    c._refresh_patch_state_list()
    assert _status(w).startswith("Reconstruction complete")


def test_pr_status_routes_through_the_one_setter(full):
    from napari_epicycle.ui_shell.theme import PANEL_ERROR_STYLE
    w, c, _statuses = full
    c._pr_status("No patch was segmented", "error")
    assert _status(w) == "No patch was segmented"
    assert w.pipeline_status_label.styleSheet() == PANEL_ERROR_STYLE
    assert c._pr_done_kind(3, 0) == "neutral"
    assert c._pr_done_kind(2, 1) == "warn"
    assert c._pr_done_kind(0, 3) == "error"


def test_pr_status_falls_back_on_a_fake_ui():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )

    class _L:
        def setText(self, t):
            self.t = t

    class _UI:
        pass

    c = C.__new__(C)
    c.ui = _UI()
    c.ui.pipeline_status_label = _L()
    c._pr_status("x")
    assert c.ui.pipeline_status_label.t == "x"
    c._refresh_pr_foot_gates()        # a fake without the setter: no-op


def test_no_controller_line_writes_the_status_label_directly():
    """Every status write goes through ``_pr_status``."""
    tree = ast.parse((FR / "controller.py").read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and node.attr in ("pipeline_status_label",
                                  "prediction_status_label",
                                  "reconstruction_status_label",
                                  "batch_progress_label")):
            bad.append(node.lineno)
    assert bad == [] or all(
        _in_function(tree, ln, "_pr_status") for ln in bad), bad


def _in_function(tree, lineno, name) -> bool:
    for fn in ast.walk(tree):
        if (isinstance(fn, ast.FunctionDef) and fn.name == name
                and fn.lineno <= lineno <= fn.end_lineno):
            return True
    return False

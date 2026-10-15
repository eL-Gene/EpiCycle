"""Brick P module half: the Curation right column and run row builders
(offscreen Qt).  Qt-backed: run this file on its own (``-o addopts=""``)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox,        # noqa: E402
                            QLabel, QLineEdit, QProgressBar, QPushButton,
                            QRadioButton, QWidget)
from superqt import QCollapsible                                        # noqa: E402

from napari_epicycle.hitl_training_annotation.ui import _step2_curation as C  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def host(qapp):
    w = QWidget()
    yield w
    w.deleteLater()


class _Clicks:
    def __init__(self):
        self.n = 0

    def hit(self, *_):
        self.n += 1


def _radios(host):
    shared = QWidget(host)
    return QRadioButton("XY", shared), QRadioButton("XZ", shared)


# ------------------------------------------------------------- run row
class TestRunRow:
    def _row(self, host):
        picker = QComboBox(host)
        btns = [QPushButton(n, host) for n in ("new", "manage", "irun", "iext")]
        counters = [_Clicks() for _ in range(3)]
        for b, c in zip(btns[1:], counters):
            b.clicked.connect(c.hit)
        row = C.build_run_row(picker, btns[0], btns[1], btns[2], btns[3],
                              QPushButton("?", host), host)
        return row, btns, counters

    def test_menu_clicks_the_three_hidden_buttons(self, host):
        row, btns, counters = self._row(host)
        row.act_manage.trigger()
        row.act_import_run.trigger()
        row.act_import_external.trigger()
        row.act_import_external.trigger()
        assert [c.n for c in counters] == [1, 1, 2]

    def test_old_buttons_hidden_and_parented(self, host):
        row, btns, _ = self._row(host)
        for b in btns[1:]:
            assert b.isHidden()
            assert b.parent() is row
        assert not btns[0].isHidden() or btns[0].parent() is not None

    def test_menu_words_and_caption_is_not_actionable(self, host):
        row, _, _ = self._row(host)
        texts = [a.text() for a in row.menu.actions() if not a.isSeparator()]
        assert texts == [C.CUR_MANAGE_RUNS_ACTION, C.CUR_MANAGE_CAPTION,
                         C.CUR_IMPORT_PAIRS_ACTION,
                         C.CUR_IMPORT_EXTERNAL_ACTION]
        assert not row.act_caption.isEnabled()

    def test_enabled_state_mirrors_hidden_buttons(self, host):
        row, btns, _ = self._row(host)
        btns[2].setEnabled(False)
        assert not row.act_import_run.isEnabled()
        assert row.act_manage.isEnabled()
        for b in btns[1:]:
            b.setEnabled(False)
        assert not row.manage_menu_button.isEnabled()
        btns[1].setEnabled(True)
        assert row.manage_menu_button.isEnabled()


# --------------------------------------------------------- view switch
class TestViewSwitch:
    def test_click_segment_clicks_radio_and_mirrors(self, host):
        lat, ax = _radios(host)
        lat.setChecked(True)
        sw = C.ViewSwitch(lat, ax, host)
        assert sw.buttons["lateral"].isChecked()
        sw.buttons["axial"].click()
        assert ax.isChecked() and not lat.isChecked()
        assert sw.buttons["axial"].isChecked()
        assert not sw.buttons["lateral"].isChecked()
        assert sw.active_bucket() == "axial"

    def test_radio_change_mirrors_to_proxy(self, host):
        lat, ax = _radios(host)
        lat.setChecked(True)
        sw = C.ViewSwitch(lat, ax, host)
        ax.setChecked(True)
        assert sw.buttons["axial"].isChecked()
        lat.setChecked(True)
        assert sw.buttons["lateral"].isChecked()
        assert not sw.buttons["axial"].isChecked()

    def test_toggled_fires_once_per_change(self, host):
        lat, ax = _radios(host)
        lat.setChecked(True)
        sw = C.ViewSwitch(lat, ax, host)
        seen = []
        ax.toggled.connect(seen.append)
        sw.buttons["axial"].click()
        sw.buttons["axial"].click()  # already on: no second toggle
        assert seen == [True]

    def test_follows_enabled_and_disabled_click_does_not_flip(self, host):
        lat, ax = _radios(host)
        lat.setChecked(True)
        sw = C.ViewSwitch(lat, ax, host)
        lat.setEnabled(False)
        ax.setEnabled(False)
        assert not sw.buttons["lateral"].isEnabled()
        assert not sw.buttons["axial"].isEnabled()
        ax.setEnabled(True)
        lat.setEnabled(True)
        assert sw.buttons["axial"].isEnabled()
        # a click that the (disabled) radio ignores must not flip the proxy
        ax.setEnabled(False)
        sw._on_axial_clicked()
        assert not ax.isChecked()
        assert sw.buttons["lateral"].isChecked()

    def test_radios_hidden_and_parented_under_switch(self, host):
        lat, ax = _radios(host)
        sw = C.ViewSwitch(lat, ax, host)
        assert lat.isHidden() and ax.isHidden()
        assert lat.parent() is sw and ax.parent() is sw


# ------------------------------------------------------ progress table
class TestProgressTable:
    def test_refresh_numbers_and_bold_active(self, host):
        t = C.build_progress_table(host)
        t.refresh({"lateral": {"predicted": 22, "curating": 3, "accepted": 2,
                               "staged": 5, "committed": 8},
                   "axial": {"predicted": 9, "curating": 4}},
                  "axial", "a note")
        assert t.cells[("predicted", "lateral")].text() == "22"
        assert t.cells[("committed", "lateral")].text() == "8"
        assert t.cells[("curating", "axial")].text() == "4"
        assert t.cells[("committed", "axial")].text() == "0"
        assert "bold" in t.cells[("predicted", "axial")].styleSheet()
        assert "normal" in t.cells[("predicted", "lateral")].styleSheet()
        assert "bold" in t.header_labels["axial"].styleSheet()
        assert "normal" in t.header_labels["lateral"].styleSheet()
        assert t.note_label.text() == "a note"

    def test_switching_active_bucket_moves_bold(self, host):
        t = C.build_progress_table(host)
        t.refresh({}, "axial", "")
        t.refresh({}, "lateral", "")
        assert "bold" in t.header_labels["lateral"].styleSheet()
        assert "normal" in t.header_labels["axial"].styleSheet()

    def test_zero_is_grey_and_garbage_is_zero(self, host):
        t = C.build_progress_table(host)
        t.refresh({"lateral": {"predicted": "x", "curating": 1}}, "lateral", "")
        assert t.cells[("predicted", "lateral")].text() == "0"
        assert C._ZERO_COLOUR in t.cells[("predicted", "lateral")].styleSheet()
        assert C._ZERO_COLOUR not in t.cells[("curating", "lateral")].styleSheet()

    def test_five_status_rows_in_order(self):
        assert [k for k, *_ in C.STATUS_ROWS] == [
            "predicted", "curating", "accepted", "staged", "committed"]
        assert [w for _, w, *_ in C.STATUS_ROWS][-1] == "Trained"

    def test_note_helper(self):
        s = C.compose_progress_note((1, 3), 4, 3)
        assert s == ("Trained = in a round model already (lateral rounds "
                     "1–3). Finalized = waiting for the next Train. "
                     "Next round: lateral 4, axial 3.")
        assert "lateral round 2)" in C.compose_progress_note((2, 2), 3, 1)
        assert "no round yet" in C.compose_progress_note(None, 0, 0)


# ------------------------------------------------------------ stage line
class TestStageLine:
    def test_ready(self):
        assert C.compose_stage_line(patch="WholeScale_patch_4",
                                    bucket="lateral", n_selected=1,
                                    n_slices=7) == (
            "Ready · WholeScale_patch_4 · lateral · "
            "1 slice selected")

    def test_plural_and_none_selected(self):
        assert C.compose_stage_line(patch="p", bucket="axial", n_selected=3,
                                    n_slices=7).endswith("3 slices selected")
        assert C.compose_stage_line(patch="p", n_selected=0,
                                    n_slices=7).endswith("no slice selected")

    def test_no_slices_in_view(self):
        assert C.compose_stage_line(patch="p", n_slices=0) == (
            "Ready · no slices in this view yet")

    def test_no_patch(self):
        assert C.compose_stage_line() == "Ready · no patch loaded"

    def test_locked_wins(self):
        assert C.compose_stage_line(locked=True, round_n=4, bucket="lateral",
                                    patch="p", busy_text="x") == (
            "Locked · round 4 (lateral) is training · cancel it "
            "in Training")

    def test_busy_beats_ready(self):
        assert C.compose_stage_line(patch="p", busy_text="Loading…") == (
            "Loading…")

    def test_inference_summary(self):
        assert C.compose_inference_summary(
            "Balanced", "membrane / nuclei", 0.4, 0.0, 30) == (
            "Balanced · membrane / nuclei · flow 0.4 · "
            "cellprob 0.0 · diameter 30 px")


# ----------------------------------------------------------- model block
def _model_block(host):
    src = QComboBox(host)
    src.addItems(["Auto (recommended)", "Round model", "External model file",
                  "Cellpose pretrained", "None (manual / SAM only)"])
    active = QLabel("x", host)
    path = QLineEdit(host)
    browse = QPushButton("Browse", host)
    pre = QComboBox(host)
    pre.addItems(["cyto3"])
    auto = QCheckBox("old", host)
    c1, c2 = QComboBox(host), QComboBox(host)
    coll = QCollapsible("Inference Parameters", host)
    blk = C.build_model_block(src, active, path, browse, pre, auto, c1, c2,
                              coll, host)
    return blk, src, path, pre, auto


class TestModelBlock:
    def test_path_row_only_for_external(self, host):
        blk, src, path, pre, auto = _model_block(host)
        assert blk.path_row.isHidden()
        src.setCurrentText("External model file")
        assert not blk.path_row.isHidden()
        src.setCurrentText("Cellpose pretrained")
        assert blk.path_row.isHidden()

    def test_greyed_reason_text(self, host):
        blk, src, path, pre, auto = _model_block(host)
        assert pre.toolTip() == C.CUR_MODEL_PRETRAINED_NOTE
        src.setCurrentText("External model file")
        assert pre.toolTip() == C.CUR_MODEL_EXTERNAL_NOTE
        assert blk.pretrained_note.text() == C.CUR_MODEL_EXTERNAL_NOTE

    def test_reworded_checkbox_and_summary(self, host):
        blk, src, path, pre, auto = _model_block(host)
        assert auto.text() == C.CUR_MODEL_AUTO_UPDATE
        blk.set_inference_summary("Balanced · flow 0.4")
        assert blk.inference_summary.text() == "Balanced · flow 0.4"

    def test_does_not_touch_enabled_state(self, host):
        blk, src, path, pre, auto = _model_block(host)
        pre.setEnabled(False)
        src.setCurrentText("External model file")
        assert not pre.isEnabled()


# ------------------------------------------------------------------ foot
def _foot(host):
    btns = [QPushButton("x", host) for _ in range(5)]
    bar = QProgressBar(host)
    foot = C.build_pinned_foot(*btns, bar, None, host)
    return foot, btns, bar


class TestFoot:
    def test_exactly_one_primary(self, host):
        foot, btns, _ = _foot(host)
        prim = [b for b in btns if b.property("epiPrimary")]
        assert prim == [btns[0]]
        assert [b.text() for b in btns] == [
            "Open 2D annotator", "Accept", "Finalize accepted", "Discard",
            "Revert"]
        assert foot.open_btn is btns[0]

    def test_stage_and_busy(self, host):
        foot, btns, bar = _foot(host)
        foot.set_stage("Ready · p")
        assert foot.stage_label.text() == "Ready · p"
        foot.set_busy("Predicting…")
        # the bar never draws text in busy mode: the words are the bar's tooltip
        assert foot.busy_label.isHidden()   # words: tooltip + stage line
        assert bar.toolTip() == "Predicting…"
        foot.set_busy(None)
        assert foot.busy_label.isHidden() and not bar.isTextVisible()
        assert bar.height() >= 0 and bar.minimumHeight() == C.FOOT_BAR_H


# --------------------------------------------------------- right column
def _column(host):
    lat, ax = _radios(host)
    lat.setChecked(True)
    sw = C.ViewSwitch(lat, ax, host)
    table = C.build_progress_table(host)
    blk, *_ = _model_block(host)
    foot, btns, bar = _foot(host)
    col = C.build_right_column(sw, table, blk, foot, host)
    return col, foot


class TestRightColumn:
    def test_every_widget_has_a_parent(self, host):
        col, foot = _column(host)
        for w in col.findChildren(QWidget) + [col, foot]:
            assert w.parent() is not None, w

    def test_foot_is_sibling_of_scroll_not_inside(self, host):
        col, foot = _column(host)
        assert foot.parent() is col
        assert not col.scroll.isAncestorOf(foot)
        assert col.scroll.isAncestorOf(col.view_switch)
        assert col.scroll.isAncestorOf(col.progress_table)
        assert col.scroll.isAncestorOf(col.model_block)

    def test_min_height_fits_the_budget(self, host):
        col, foot = _column(host)
        col.resize(300, 800)
        # The column's foot ~150 px + the scroll floor 120 px must
        # fit the ~290 px the open section gets (the run row sits above)
        assert col.scroll.minimumHeight() == C.SCROLL_MIN_H
        assert foot.minimumSizeHint().height() <= 160
        assert col.minimumSizeHint().height() <= 290 - 32

    def test_foot_stays_visible_when_squeezed(self, host):
        col, foot = _column(host)
        col.resize(300, col.minimumSizeHint().height())
        col.show()
        QApplication.processEvents()
        assert foot.geometry().bottom() <= col.height()
        assert foot.height() >= foot.minimumSizeHint().height()
        col.hide()

    def test_no_horizontal_scrollbar_at_578_column_width(self, host):
        col, foot = _column(host)
        col.resize(330, 400)
        col.show()
        QApplication.processEvents()
        assert not col.scroll.horizontalScrollBar().isVisible()
        col.hide()


# ===========================================================================
# Brick P integration half: the real main_widget hookup + the controller
# seams (c), (d), hook 5.
# ===========================================================================
import inspect                                                          # noqa: E402

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (  # noqa: E402,E501
    TrainingAnnotationController as Ctl)


def _real_widget(qapp):
    """The Step 2 widget the layout tests' way: no controller."""
    from napari_epicycle.hitl_training_annotation.ui.main_widget import (
        TrainingAnnotationHITL2D)

    w = TrainingAnnotationHITL2D.__new__(TrainingAnnotationHITL2D)
    QWidget.__init__(w)
    w.viewer = None
    w.setLayout(w._create_main_layout())
    w._apply_theme()
    return w


#: The Curation attributes of the wiring map (section 4.2) with the type
#: each must keep.
_CURATION_ATTRS = {
    "run_picker_combo": QComboBox, "new_run_button": QPushButton,
    "import_pairs_button": QPushButton, "import_external_button": QPushButton,
    "manage_runs_button": QPushButton, "patch_busy_bar": QProgressBar,
    "launch_multiview_button": QPushButton,
    "accept_annotation_button": QPushButton,
    "reject_annotation_button": QPushButton,
    "stage_current_view_button": QPushButton,
    "unstage_selected_button": QPushButton, "view_results_button": QPushButton,
    "finalize_labels_button": QPushButton,
    "filter_all_btn": QPushButton, "filter_predicted_btn": QPushButton,
    "filter_curating_btn": QPushButton, "filter_accepted_btn": QPushButton,
    "filter_staged_btn": QPushButton, "filter_committed_btn": QPushButton,
    "get_view_lateral": QRadioButton, "get_view_axial": QRadioButton,
    "curation_progress_container": QWidget,
    "model_source_combo": QComboBox, "imported_model_path": QLineEdit,
    "imported_model_browse_button": QPushButton,
    "curation_pretrained_combo": QComboBox, "auto_update_model_cb": QCheckBox,
    "apply_model_source_button": QPushButton,
    "active_model_label": QLabel, "active_model_summary_value": QLabel,
    "active_model_source_value": QLabel, "active_lateral_model_value": QLabel,
    "active_axial_model_value": QLabel, "_inference_collapsible": QCollapsible,
    "inference_preset_combo": QComboBox, "inference_channel1_combo": QComboBox,
    "inference_channel2_combo": QComboBox,
    "apply_inference_params_button": QPushButton,
    "slice_summary_label": QLabel, "summary_round_value": QLabel,
    "summary_staged_value": QLabel, "review_tabs": QWidget,
    "queue_filter_combo": QComboBox,
}


class TestRealWidgetHookup:
    @pytest.fixture(scope="class")
    def widget(self, qapp):
        return _real_widget(qapp)

    def test_every_wiring_map_curation_attribute_survives(self, widget):
        for name, typ in _CURATION_ATTRS.items():
            assert isinstance(getattr(widget, name, None), typ), name
        for name in ("train_patch_list", "segment_slice_data_list",
                     "accepted_queue_list", "staged_queue_list",
                     "sample_queue_list"):
            assert hasattr(getattr(widget, name), "selectedItems"), name
        assert widget.accepted_queue_list is widget.segment_slice_data_list
        assert (widget.curation_progress_container
                is widget.curation_progress_table)
        assert (widget.curation_progress_layout
                is widget.curation_progress_table.layout())

    def test_the_foot_has_one_primary_and_the_asked_words(self, widget):
        foot = widget.curation_foot
        assert foot.open_btn is widget.launch_multiview_button
        assert widget.launch_multiview_button.text() == "Open 2D annotator"
        assert widget.launch_multiview_button.property("epiPrimary") is True
        assert [b.property("epiPrimary") for b in foot.buttons] == [False] * 4
        assert [b.text() for b in foot.buttons] == [
            "Accept", "Finalize accepted", "Discard", "Revert"]
        assert foot.buttons == (
            widget.accept_annotation_button, widget.stage_current_view_button,
            widget.reject_annotation_button, widget.unstage_selected_button)
        # _apply_theme must not have restyled the foot
        assert widget.accept_annotation_button.styleSheet() == C.SECONDARY_BTN_STYLE

    def test_the_busy_bar_lives_in_the_foot(self, widget):
        bar = widget.patch_busy_bar
        assert widget.curation_foot.busy_bar is bar
        assert widget.curation_foot.isAncestorOf(bar)
        assert widget.curation_foot.isAncestorOf(widget.curation_stage_label)
        assert (bar.minimum(), bar.maximum(), bar.value()) == (0, 100, 0)

    def test_the_radios_are_hidden_and_the_proxy_mirrors_them(self, widget):
        sw = widget.curation_view_switch
        assert widget.get_view_lateral.isHidden()
        assert widget.get_view_axial.isHidden()
        assert widget.get_view_lateral.parent() is widget.get_view_axial.parent()
        assert widget.get_view_lateral.isChecked()
        assert sw.buttons["lateral"].isChecked()
        sw.buttons["axial"].click()
        assert widget.get_view_axial.isChecked()
        assert not widget.get_view_lateral.isChecked()
        widget.get_view_lateral.setChecked(True)
        assert sw.buttons["lateral"].isChecked()
        assert not sw.buttons["axial"].isChecked()
        widget.get_view_axial.setEnabled(False)
        assert not sw.buttons["axial"].isEnabled()
        widget.get_view_axial.setEnabled(True)

    def test_the_launch_button_and_actions_left_the_left_column(self, widget):
        left = widget.curation_splitter.widget(0)
        assert not left.isAncestorOf(widget.launch_multiview_button)
        assert not left.isAncestorOf(widget.accept_annotation_button)
        right = widget.curation_splitter.widget(1)
        assert right is widget.curation_right_column
        assert right.isAncestorOf(widget.launch_multiview_button)
        assert right.isAncestorOf(widget.curation_view_switch)
        assert (widget.curation_splitter.handleWidth(),
                left.minimumWidth(), right.minimumWidth()) == (6, 240, 290)

    def test_the_run_row_keeps_the_three_buttons_hidden_and_new_run_shown(
            self, widget):
        row = widget.curation_run_row
        assert widget.new_run_button.isHidden() is False
        for b in (widget.manage_runs_button, widget.import_pairs_button,
                  widget.import_external_button):
            assert b.isHidden()
            assert row.isAncestorOf(b)
        assert widget.new_run_button.text() == "New Run"

    def test_the_channel_combos_left_the_disclosure_for_the_model_block(
            self, widget):
        block = widget.curation_model_block
        assert block.isAncestorOf(widget.inference_channel1_combo)
        assert not widget.curation_inference_group.isAncestorOf(
            widget.inference_channel1_combo)

    def test_the_inference_summary_follows_the_knobs(self, widget):
        widget.inference_flow_thresh.setValue(0.5)
        txt = widget.curation_model_block.inference_summary.text()
        assert "flow 0.5" in txt and "cellprob 0.0" in txt
        assert txt.startswith(widget.inference_preset_combo.currentText())
        assert "diameter auto" in txt

    def test_guides_are_the_real_files(self, widget):
        from pathlib import Path
        d = Path(C._GUIDES_DIR)
        for g in (C.GUIDE_RUN, C.GUIDE_VIEW, C.GUIDE_PROGRESS, C.GUIDE_MODEL):
            assert (d / g).is_file(), g
        assert C.GUIDE_RUN == "step2_round_run.md"

    def test_stage_line_busy_and_lock(self, widget):
        class Fake:
            loaded_patch_name = "patch_4"
        widget.controller = Fake()
        try:
            widget.refresh_stage_line()
            assert widget.curation_stage_label.text() in (
                "Ready · patch_4 · lateral · no slice selected",
                "Ready · no slices in this view yet")
            widget.on_busy("Loading patch_4…")
            assert widget.curation_stage_label.text() == "Loading patch_4…"
            assert widget.patch_busy_bar.toolTip() == "Loading patch_4…"
            widget.on_busy(None)
            assert widget.patch_busy_bar.toolTip() == ""
            widget.on_training_locked(True, "Cell_lateral_round_03", "lateral")
            assert not widget.curation_lock_line.isHidden()
            assert widget.curation_stage_label.text() == (
                "Locked · round 3 (lateral) is training · cancel it in "
                "Training")
            widget.on_training_locked(False, "", "")
            assert widget.curation_lock_line.isHidden()
        finally:
            del widget.controller


# ---------------------------------------------------------- controller seams
class _FakeTable:
    def __init__(self):
        self.calls = []

    def refresh(self, counts, active, note):
        self.calls.append((counts, active, note))


class _FakeUI:
    def __init__(self, table=None):
        if table is not None:
            self.curation_progress_table = table
        self.stage_calls = 0

    def refresh_stage_line(self):
        self.stage_calls += 1


def _ctrl(ui, view="XY"):
    c = Ctl.__new__(Ctl)
    c.ui = ui
    c.selected_view = view
    c.loaded_patch_name = "p"
    c.current_round_by_view = {"lateral": 3, "axial": 2}
    c.sample_records = {
        "a": {"status": "predicted", "bucket": "lateral"},
        "b": {"status": "staged", "bucket": "lateral"},
        "c": {"status": "committed", "bucket": "lateral"},
        "d": {"status": "committed", "bucket": "lateral"},
        "e": {"status": "curating", "bucket": "axial"},
        "f": {"status": "weird", "bucket": "axial"},
    }
    c._refresh_patch_list_badges = lambda: None
    return c


class TestSeamC:
    def test_refresh_fills_the_table_from_the_records(self):
        t = _FakeTable()
        _ctrl(_FakeUI(t))._refresh_sample_queue()
        (counts, active, note), = t.calls
        assert counts["lateral"] == {"predicted": 1, "staged": 1,
                                     "committed": 2}
        assert counts["axial"] == {"curating": 1}
        assert active == "lateral"
        assert "lateral rounds 0–2" in note
        assert "Next round: lateral 3, axial 2" in note

    def test_the_active_bucket_follows_the_view(self):
        t = _FakeTable()
        _ctrl(_FakeUI(t), view="XZ")._refresh_sample_queue()
        assert t.calls[0][1] == "axial"

    def test_round_zero_says_no_round_yet(self):
        t = _FakeTable()
        c = _ctrl(_FakeUI(t))
        c.current_round_by_view = {"lateral": 0, "axial": 0}
        c._refresh_sample_queue()
        assert "no round yet" in t.calls[0][2]

    def test_a_ui_without_the_table_still_runs_the_seam(self):
        ui = _FakeUI(None)
        _ctrl(ui)._refresh_sample_queue()
        assert ui.stage_calls == 1

    def test_a_failing_table_does_not_break_the_refresh(self):
        class Bad(_FakeTable):
            def refresh(self, *a):
                raise RuntimeError("boom")

        ui = _FakeUI(Bad())
        _ctrl(ui)._refresh_sample_queue()
        assert ui.stage_calls == 1

    def test_the_old_rebuild_block_is_gone(self):
        src = inspect.getsource(Ctl._refresh_sample_queue)
        assert "deleteLater" not in src
        assert "progress_layout" not in src
        assert "setStyleSheet" not in src


class TestSeamDStageLine:
    def test_refresh_sample_queue_and_jump_call_it(self):
        ui = _FakeUI()
        c = _ctrl(ui)
        c._refresh_sample_queue()
        assert ui.stage_calls == 1
        c._parse_queue_selected_sample_id = lambda: None
        c._jump_to_selected_queue_item()
        assert ui.stage_calls == 2
        c._parse_queue_selected_sample_id = lambda: "a"
        c.focus_sample = lambda sid: None
        c._jump_to_selected_queue_item()
        assert ui.stage_calls == 3

    def test_a_ui_without_the_method_or_a_failing_one_is_fine(self):
        _ctrl(object())._refresh_curation_stage_line()

        class Bad:
            def refresh_stage_line(self):
                raise RuntimeError("boom")

        _ctrl(Bad())._refresh_curation_stage_line()

    def test_the_active_model_line_uses_the_text_constants(self):
        src = inspect.getsource(Ctl._refresh_curation_model_status)
        assert "CUR_MODEL_ACTIVE_LINE" in src

    def test_the_active_model_line_values(self):
        class Lbl:
            def __init__(self):
                self.t = None

            def setText(self, t):
                self.t = t

            def setToolTip(self, t):
                self.tt = t

        class UI:
            active_model_summary_value = Lbl()

        c = _ctrl(UI())
        c.active_curation_checkpoints = {
            "lateral": "/ws/Cell_lateral_round_02.model", "axial": ""}
        c.current_config = {"model_type": "cyto3"}
        c.curation_model_source_mode = "auto"
        c.imported_curation_model_path = ""
        c._refresh_curation_model_status()
        assert UI.active_model_summary_value.t == (
            "Active: lat: Cell · round_02  |  ax: cyto3")
        assert "Cell_lateral_round_02.model" in UI.active_model_summary_value.tt
        c.curation_model_source_mode = "imported"
        c.imported_curation_model_path = "/x/nuclei_v1_axial_round_02.model"
        c._refresh_curation_model_status()
        assert UI.active_model_summary_value.t == (
            "Active: lat + ax: nuclei_v1_axial_round_02.model (file)")

    def _active(self, mode, lat="", imported="", pre="cyto3"):
        class Lbl:
            t = None

            def setText(self, t):
                self.t = t

            def setToolTip(self, t):
                pass

        class UI:
            active_model_summary_value = Lbl()

        ui = UI()
        c = _ctrl(ui)
        c.active_curation_checkpoints = {"lateral": lat, "axial": ""}
        c.current_config = {"model_type": "cyto3"}
        c.curation_model_source_mode = mode
        c.imported_curation_model_path = imported
        c.curation_pretrained_model_type = pre
        c._refresh_curation_model_status()
        return ui.active_model_summary_value.t

    def test_active_line_pretrained_with_a_round_names_the_pretrained(self):
        assert self._active(
            "pretrained", lat="/w/Cell_lateral_round_02.model", pre="cyto2"
        ) == "Active: lat: cyto2  |  ax: cyto2"

    def test_active_line_auto_with_a_file_and_no_round_names_the_file(self):
        t = self._active("auto", imported="/x/mine.model")
        assert t == "Active: lat: mine  |  ax: mine"

    def test_active_line_round_mode_without_a_round_is_none(self):
        assert self._active("round") == "Active: lat: none  |  ax: none"

    def test_active_line_round_mode_names_the_round_per_bucket(self):
        assert self._active(
            "round", lat="/w/Cell_lateral_round_02.model"
        ) == "Active: lat: Cell · round_02  |  ax: none"


class TestHook5Busy:
    def _c(self, ui):
        c = _ctrl(ui)
        c._append_ui_log = lambda text: None
        return c

    def test_begin_and_end_call_on_busy(self):
        calls = []

        class UI:
            def on_busy(self, text):
                calls.append(text)

        c = self._c(UI())
        c._begin_busy("Loading patch…", cursor=False)
        c._begin_busy("inner", cursor=False)       # nested: no second call
        c._end_busy()
        assert calls == ["Loading patch…"]
        c._end_busy()
        assert calls == ["Loading patch…", None]

    def test_no_message_means_no_words(self):
        calls = []

        class UI:
            def on_busy(self, text):
                calls.append(text)

        c = self._c(UI())
        c._begin_busy("", cursor=False)
        assert calls == [None]
        c._end_busy()

    def test_a_ui_without_the_hook_or_a_failing_one_is_fine(self):
        c = self._c(object())
        c._begin_busy("x", cursor=False)
        c._end_busy()

        class Bad:
            def on_busy(self, text):
                raise RuntimeError("boom")

        c = self._c(Bad())
        c._begin_busy("x", cursor=False)
        c._end_busy()


# ------------------------------------------------- audit fixes (room, seams)
class TestRoom:
    @pytest.fixture(scope="class")
    def widget(self, qapp):
        return _real_widget(qapp)

    def test_i6_the_columns_fit_the_218_px_under_the_run_row(self, widget):
        right = widget.curation_right_column
        left = widget.curation_splitter.widget(0)
        assert right.minimumSizeHint().height() <= 218
        # Step 3's own button sizes (polish D-A): 32 px primary, 26 px
        # secondaries -- the foot was 110 with the satellite-dock buttons
        assert widget.curation_foot.minimumSizeHint().height() <= 124
        assert left.minimumSizeHint().height() <= 218,             left.minimumSizeHint().height()

    def test_the_foot_is_pinned_outside_the_scroll_area(self, widget):
        right = widget.curation_right_column
        assert not right.scroll.isAncestorOf(widget.curation_foot)
        # 80 px in a tall dock, 48 in a short one (polish: the snapped
        # viewport may shrink to the switch + the counts)
        assert right.scroll.minimumHeight() in (C.SCROLL_MIN_H,
                                                C.SCROLL_MIN_H_SHORT)

    def test_short_mode_turns_descs_into_header_tooltips(self, widget):
        widget.resize(578, 576)
        widget._apply_curation_room()
        right = widget.curation_right_column
        for header, desc in right._descs:
            assert desc.isHidden()
            assert header.title_label.toolTip() == desc.text()
        for header, desc in widget._left_descs:
            assert desc.isHidden() and header.toolTip() == desc.text()
        widget.resize(578, 800)
        widget._apply_curation_room()
        for header, desc in right._descs:
            assert not desc.isHidden() or not widget.isVisible()
            assert header.title_label.toolTip() == ""

    def test_the_left_column_is_a_vertical_splitter_of_two_panes(self, widget):
        sp = widget.curation_left_splitter
        assert sp.count() == 2
        assert sp.widget(0).isAncestorOf(widget.train_patch_list)
        assert sp.widget(1).isAncestorOf(widget.segment_slice_data_list)

    def test_the_use_popup_is_themed_on_the_root_sheet(self, widget):
        assert "QComboBox QAbstractItemView" in widget.styleSheet()
        assert "#2a2d33" in widget.styleSheet()


class TestSeamG:
    def test_open_never_loads_a_selected_row(self, qapp):
        from qtpy.QtWidgets import QListWidget
        lst = QListWidget()
        lst.addItem("patch_A.tiff")
        lst.addItem("patch_B.tiff")
        lst.item(1).setSelected(True)

        class UI:
            train_patch_list = lst

        c = _ctrl(UI())
        c.loaded_patch_name = "patch_A"
        calls = []
        c._ensure_runtime_defaults = lambda: None
        c._ensure_llm_removal_callback = lambda: None
        c._effective_stack_path = lambda: "/ws/patches"
        c._load_patch = lambda *a, **k: calls.append(("load", a, k))
        c._select_patch_item = lambda *a, **k: calls.append(("select", a))
        try:
            c._launch_multiview_annotator()
        except Exception:
            pass     # the rest of a bare controller may fail; not our seam
        assert calls == []

    def test_the_source_no_longer_reads_the_row_text(self):
        src = inspect.getsource(Ctl._launch_multiview_annotator)
        assert "selected_items[0].text()" not in src
        assert "_load_patch(force_axis" not in src


class TestShutdownOrder:
    def test_the_hook_runs_after_the_abort_flag(self):
        src = inspect.getsource(Ctl.shutdown)
        assert src.index("_abort.set()") < src.index('"on_shutdown"')

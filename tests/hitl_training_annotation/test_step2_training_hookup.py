"""Brick T (integration half): the Training section on the REAL
``TrainingAnnotationHITL2D`` (offscreen Qt, no controller) -- the attributes
and their types, Train . Cancel as two buttons with one primary, the foot
splitter, the Loss collapsible closed idle / open on ``on_train_started`` /
closed on ``on_train_finished``, the lock line and the per-widget greying of
``on_training_locked`` (merged with Curation's and Session's), the 218 px
room.  Run on its own with ``-o addopts=""``."""
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox,  # noqa: E402
                            QDoubleSpinBox, QLabel, QLineEdit, QProgressBar,
                            QPushButton, QRadioButton, QSpinBox, QWidget)

from napari_epicycle.hitl_training_annotation.ui import _step2_text as T  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui import _step2_training as TR  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui.loss_plot_widget import LossPlotWidget  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


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


_TRAINING_ATTRS = {
    "update_and_train_button": QPushButton, "train_cancel_btn": QPushButton,
    "training_progress_bar": QProgressBar, "training_stage_label": QLabel,
    "dataset_staging": QLabel, "model_name_lateral_label": QLabel,
    "model_name_axial_label": QLabel, "training_mode_combo": QComboBox,
    "model_selector": QComboBox, "starting_checkpoint_path": QLineEdit,
    "starting_checkpoint_browse": QPushButton, "starting_checkpoint_status": QLabel,
    "_init_hint_label": QLabel, "epochs_sb": QSpinBox, "batch_sb": QSpinBox,
    "device_gpu": QRadioButton, "device_cpu": QRadioButton,
    "lr_db": QDoubleSpinBox, "wd_db": QDoubleSpinBox, "minmask_sb": QSpinBox,
    "chan1_combo": QComboBox, "chan2_combo": QComboBox,
    "norm_cb": QCheckBox, "rescale_cb": QCheckBox, "flow_cb": QCheckBox,
    "save_and_set_config_button": QPushButton, "load_config_button": QPushButton,
    "training_history_button": QPushButton, "loss_plot": LossPlotWidget,
}


@pytest.fixture(scope="class")
def widget(qapp):
    return _real_widget(qapp)


class TestAttributes:
    def test_every_attribute_has_its_type(self, widget):
        for name, typ in _TRAINING_ATTRS.items():
            assert isinstance(getattr(widget, name), typ), name
        assert widget.running_training_loss_plot is widget.loss_plot

    def test_every_widget_has_a_parent(self, widget):
        for name in _TRAINING_ATTRS:
            assert getattr(widget, name).parent() is not None, name
        assert not [w for w in widget.training_panel.findChildren(QWidget)
                    if w.parent() is None]

    def test_the_lock_list_widgets_stay_literal_assignments(self):
        src = Path(__import__(
            "napari_epicycle.hitl_training_annotation.ui.main_widget",
            fromlist=["x"]).__file__).read_text(encoding="utf-8")
        for name in ("epochs_sb", "starting_checkpoint_path",
                     "starting_checkpoint_browse", "training_mode_combo",
                     "update_and_train_button", "train_cancel_btn"):
            assert re.search(rf"self\.{name}\s*=", src), name

    def test_the_mode_words_and_the_advanced_words(self, widget):
        assert [widget.training_mode_combo.itemText(i) for i in range(2)] == [
            T.MODE_CONTINUE, T.MODE_SCRATCH]
        labels = {lab.text() for lab in widget.advanced_params_group.findChildren(QLabel)}
        for word in (T.ADV_MIN_OBJECTS, T.ADV_CHANNEL2, T.ADV_LEARNING_RATE,
                     T.ADV_WEIGHT_DECAY, "learning_rate", "weight_decay",
                     "min_train_masks"):
            assert word in labels, word
        assert widget.flow_cb.text() == T.ADV_FLOWS
        for old in ("Min mask size:", "Auxiliary channel (optional):", "Compute flows"):
            assert old not in labels
        assert widget._training_adv_collapsible.text() == T.ADVANCED_TITLE

    def test_advanced_starts_closed(self, widget):
        assert not widget._training_adv_collapsible.isExpanded()

    def test_the_loss_view_words_follow_the_spec(self, widget):
        combo = widget.loss_plot.mode_combo
        assert [combo.itemText(i) for i in range(3)] == list(T.LOSS_MODE_WORDS)
        assert widget.training_foot.loss.view_combo is combo


class TestFoot:
    def test_two_distinct_buttons_one_primary(self, widget):
        tr, ca = widget.update_and_train_button, widget.train_cancel_btn
        assert tr is not ca
        assert tr.text() == "Train" and ca.text() == "Cancel"
        assert tr.styleSheet() == TR.PRIMARY_BTN_STYLE
        assert ca.styleSheet() != tr.styleSheet()
        assert tr.isEnabled() and not ca.isEnabled()

    def test_the_foot_sits_under_a_vertical_splitter(self, widget):
        p = widget.training_panel
        assert p.splitter.orientation() == Qt.Vertical
        assert p.splitter.indexOf(p.foot) == 1
        assert p.splitter.indexOf(p.columns_holder) == 0
        assert p.columns_holder.isAncestorOf(p.columns_splitter)
        assert not p.splitter.childrenCollapsible()
        assert p.columns_splitter.orientation() == Qt.Horizontal

    def test_the_idle_stage_line_is_the_bare_sentence(self, widget):
        assert widget.training_stage_label.text() == T.STAGE_IDLE_BARE


class TestHooks:
    def test_loss_is_closed_idle_opens_on_start_closes_on_finish(self, qapp):
        w = _real_widget(qapp)
        loss = w.training_foot.loss
        assert not loss.is_open()
        w.on_train_started("lateral", "Cell_lateral_round_04")
        assert loss.is_open() and not loss.body.isHidden()
        assert w.loss_plot._live_family == "lateral"
        assert not w.update_and_train_button.isEnabled()
        assert w.train_cancel_btn.isEnabled()
        assert loss.caption_label.text().startswith("lat r4")
        assert w.training_stage_label.text() == (
            "Cell · lateral round 4 · starting…")
        w.on_training_epoch(412, 1000, 0.81)
        assert w.training_stage_label.text() == (
            "Cell · lateral round 4 · epoch 412 / 1000 · loss 0.81")
        assert loss.caption_label.text() == "lat r4 · epoch 412 · 0.81"
        w.on_train_finished()
        assert not loss.is_open() and loss.body.isHidden()

    def test_the_lock_line_and_the_merged_lock_lines(self, qapp):
        w = _real_widget(qapp)
        panel = w.training_panel
        assert panel.lock_line.isHidden()
        w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
        assert panel.lock_line.text() == T.TRAIN_LOCK_LINE.format(
            n=3, bucket="lateral")
        assert not panel.lock_line.isHidden()
        # Curation's and Session's halves of the same hook still run
        assert w.session_lock_line.text() == T.SESSION_LOCK_LINE.format(
            n=3, bucket="lateral")
        assert w.curation_lock_line.text() == T.CUR_LOCK_LINE.format(
            n=3, bucket="lateral")
        w.on_training_locked(False, "Cell_lateral_round_03", "lateral")
        assert panel.lock_line.isHidden()
        assert w.session_lock_line.text() == ""

    def test_the_setup_is_greyed_per_widget_and_restored_to_its_own_state(self, qapp):
        w = _real_widget(qapp)
        w.chan2_combo.setEnabled(False)          # its OWN prior state
        w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
        for name in ("batch_sb", "device_gpu", "device_cpu", "lr_db", "wd_db",
                     "minmask_sb", "chan1_combo", "chan2_combo", "norm_cb",
                     "rescale_cb", "flow_cb", "save_and_set_config_button",
                     "load_config_button"):
            assert not getattr(w, name).isEnabled(), name
        # greyed per widget, never a container: Advanced and the ? stay live
        assert w._training_adv_collapsible.isEnabled()
        assert w.training_history_button.isEnabled()
        w.on_training_locked(False, "Cell_lateral_round_03", "lateral")
        assert w.batch_sb.isEnabled() and w.lr_db.isEnabled()
        assert not w.chan2_combo.isEnabled()      # restored to its own state

    def test_the_staged_list_is_dimmed_under_the_lock(self, qapp):
        from napari_epicycle.hitl_training_annotation.ui import _step2_lists as L
        w = _real_widget(qapp)
        w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
        assert w.data_to_add_list.property(L.LOCK_PROPERTY) is True
        w.on_training_locked(False, "Cell_lateral_round_03", "lateral")
        assert w.data_to_add_list.property(L.LOCK_PROPERTY) is False

    def test_the_override_grey_survives_a_lock_cycle(self, qapp):
        w = _real_widget(qapp)
        col = w.training_setup_column
        col.set_hint("override", "x")
        assert not w.model_selector.isEnabled()
        w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
        # the controller re-enables its lock list on unlock, then the hook runs
        w.starting_checkpoint_path.setEnabled(True)
        w.starting_checkpoint_browse.setEnabled(True)
        w.on_training_locked(False, "Cell_lateral_round_03", "lateral")
        for x in (w.model_selector, w.starting_checkpoint_path,
                  w.starting_checkpoint_browse):
            assert not x.isEnabled()
        col.set_hint("apply", "y")
        for x in (w.model_selector, w.starting_checkpoint_path,
                  w.starting_checkpoint_browse):
            assert x.isEnabled()

    def test_an_override_flip_during_the_lock_is_not_lost(self, qapp):
        w = _real_widget(qapp)
        col = w.training_setup_column
        col.set_hint("override", "x")
        w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
        col.set_hint("apply", "y")                # flips while locked
        w.starting_checkpoint_path.setEnabled(True)
        w.starting_checkpoint_browse.setEnabled(True)
        w.on_training_locked(False, "Cell_lateral_round_03", "lateral")
        assert w.model_selector.isEnabled()


class TestRoom:
    @pytest.fixture(scope="class")
    def room_widget(self, qapp):
        return _real_widget(qapp)

    def test_the_closed_panel_fits_the_218_px_room(self, room_widget):
        p = room_widget.training_panel
        assert not p.foot.loss.is_open()
        assert p.minimumSizeHint().height() <= 218, p.minimumSizeHint().height()

    def test_the_columns_are_in_scroll_areas_over_the_pinned_foot(self, room_widget):
        p = room_widget.training_panel
        assert p.columns_splitter.minimumHeight() == TR.SPLIT_TOP_MIN_H
        assert not p.columns_splitter.isAncestorOf(p.foot)
        assert not p.columns_splitter.isAncestorOf(p.lock_line)
        for col in (p.staged_column, p.setup_column):
            assert any(s.isAncestorOf(col) for s in
                       p.columns_splitter.findChildren(TR._Scroll))

    def test_no_horizontal_scrollbar_at_578(self, room_widget):
        room_widget.resize(578, 576)
        p = room_widget.training_panel
        assert p.minimumSizeHint().width() <= 578
        for s in p.columns_splitter.findChildren(TR._Scroll):
            assert s.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff

    def test_opening_loss_grows_the_foot_not_the_columns_floor(self, room_widget):
        p = room_widget.training_panel
        closed = p.foot.minimumSizeHint().height()
        p.foot.loss.toggle_btn.setChecked(True)
        try:
            assert p.foot.minimumSizeHint().height() >= closed + TR.LOSS_MIN_H
        finally:
            p.foot.loss.toggle_btn.setChecked(False)
